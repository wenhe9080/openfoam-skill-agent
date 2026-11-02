"""mesh_export.py — OpenFOAM polyMesh 网格保存/导出模块（纯标准库，无外部依赖）

功能（生成网格后自动调用）：
  1) polyMesh_snapshot/  — 原生 polyMesh 完整快照（points/faces/owner/neighbour/boundary/cellZones/sets）
  2) mesh.tar.gz         — 打包快照 + 报告，便于下载/存档
  3) mesh.obj            — 外表面三角化（任意 3D 软件可看）
  4) mesh.msh            — Gmsh 格式（节点 + 表面面片 + 体积 hex + 物理组；体积节点顺序为几何近似，见报告说明）
  5) mesh_report.json    — 网格统计 / 质量指标 / patch / zone / 文件清单 / 下载说明

入口：
    from mesh_export import export_mesh
    manifest = export_mesh(case_dir)   # 返回 {file: size_bytes, ...} + _report
说明：每个导出动作相互独立、失败降级（不影响其它），保证「保存下来」总是成功。
"""
import json
import os
import re
import shutil
import tarfile
import datetime

# ---------- polyMesh 基础解析 ----------

def _after_int_block(txt):
    """跳过 FoamFile 头，取第一个 'N\\n(' 之后的内容。返回 (N, body)。"""
    m = re.search(r"(?m)^\s*(\d+)\s*\n\(\s*\n?", txt)
    if not m:
        return None, txt
    return int(m.group(1)), txt[m.end():]


def _read(fp):
    try:
        return open(fp, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""


def _parse_points(case_dir):
    txt = _read(os.path.join(case_dir, "constant/polyMesh/points"))
    n, body = _after_int_block(txt)
    coords = re.findall(
        r"\(\s*(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s*\)", body)
    pts = [[float(a), float(b), float(c)] for a, b, c in coords]
    return (pts[:n] if n else pts)


def _parse_faces(case_dir):
    txt = _read(os.path.join(case_dir, "constant/polyMesh/faces"))
    n, body = _after_int_block(txt)
    faces = []
    for m in re.finditer(r"(\d+)\s*\(([^)]*)\)", body):
        verts = [int(x) for x in m.group(2).split()]
        faces.append(verts)
    return (faces[:n] if n else faces)


def _parse_owner(case_dir):
    txt = _read(os.path.join(case_dir, "constant/polyMesh/owner"))
    n, body = _after_int_block(txt)
    vals = [int(x) for x in re.findall(r"-?\d+", body)]
    return (vals[:n] if n else vals)


def _parse_boundary(case_dir):
    txt = _read(os.path.join(case_dir, "constant/polyMesh/boundary"))
    patches = []
    for m in re.finditer(r"(\w+)\s*\{\s*type\s+(\w+)\s*;(.*?)\}", txt, re.S):
        name, typ, rest = m.group(1), m.group(2), m.group(3)
        if name in ("FoamFile",):
            continue
        nf = re.search(r"nFaces\s+(\d+)", rest)
        st = re.search(r"start\s+(\d+)", rest)
        patches.append({
            "name": name, "type": typ,
            "nFaces": int(nf.group(1)) if nf else 0,
            "start": int(st.group(1)) if st else 0,
        })
    return patches


def _parse_cellzones(case_dir):
    txt = _read(os.path.join(case_dir, "constant/polyMesh/cellZones"))
    if not txt.strip():
        return []
    zones = []
    for m in re.finditer(
            r"(\w+)\s*\{\s*type\s+cellZone\s*;\s*cellLabels\s+List<label>\s*\n?\s*"
            r"(\d+)\s*\n?\(\s*([0-9\s]*)\)", txt, re.S):
        name = m.group(1)
        if name in ("FoamFile", "meta"):
            continue
        labels = [int(x) for x in m.group(3).split()]
        zones.append({"name": name, "count": int(m.group(2)), "labels": labels})
    return zones


def _mesh_quality(case_dir):
    """从 log.checkMesh 提取质量指标（OpenFOAM 数值常带尾部句号，需剥离）。"""
    txt = _read(os.path.join(case_dir, "log.checkMesh"))
    q = {}
    pats = {
        "cells": r"cells:\s+(\d+)",
        "points": r"points:\s+(\d+)",
        "faces": r"faces:\s+(\d+)",
        "internal_faces": r"internal faces:\s+(\d+)",
        "non_ortho_max": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
        "non_ortho_avg": r"average:\s*([0-9.eE+-]+)",
        "skewness_max": r"Max skewness = ([0-9.eE+-]+)",
        "aspect_max": r"Max aspect ratio = ([0-9.eE+-]+)",
        "total_volume": r"Total volume = ([0-9.eE+-]+)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        if not m:
            continue
        v = m.group(1).strip().rstrip(".")      # 剥离尾部句号
        try:
            q[k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
        except ValueError:
            continue
    q["mesh_ok"] = "Mesh OK" in txt
    return q


# ---------- 生成器 ----------

def _write_snapshot(case_dir, out_dir):
    """拷贝原生 polyMesh 到 out_dir/polyMesh_snapshot/，附 manifest。"""
    src = os.path.join(case_dir, "constant", "polyMesh")
    dst = os.path.join(out_dir, "polyMesh_snapshot")
    os.makedirs(dst, exist_ok=True)
    copied = []
    for f in os.listdir(src):
        sp = os.path.join(src, f)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(dst, f))
            copied.append(f)
    if os.path.isdir(os.path.join(src, "sets")):
        shutil.copytree(os.path.join(src, "sets"), os.path.join(dst, "sets"),
                       dirs_exist_ok=True)
        copied.append("sets/")
    return copied


def _write_obj(case_dir, out_dir, faces, points, patches):
    """外表面三角化 → .obj（含 patch 分色分组）。"""
    bound_set = set()
    patch_of_face = {}
    for p in patches:
        for i in range(p["start"], p["start"] + p["nFaces"]):
            bound_set.add(i)
            patch_of_face[i] = p["name"]

    lines = ["# ChatCFD 外表面网格（由 polyMesh boundary 生成）",
             "# units: m, 顶点 0-based+1 = OpenFOAM point index"]
    for i, c in enumerate(points):
        lines.append(f"v {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}")
    lines.append("")
    cur_patch = None
    nf = 0
    for fi in range(len(faces)):
        if fi not in bound_set:
            continue
        v = faces[fi]
        if len(v) < 3:
            continue
        pn = patch_of_face[fi]
        if pn != cur_patch:
            if cur_patch is not None:
                lines.append("")
            lines.append(f"# patch {pn}")
            cur_patch = pn
        if len(v) == 4:
            lines.append(f"f {v[0]+1} {v[1]+1} {v[2]+1}")
            lines.append(f"f {v[0]+1} {v[2]+1} {v[3]+1}")
            nf += 1
    obj_path = os.path.join(out_dir, "mesh.obj")
    open(obj_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return nf


def _cell_corner_nodes(case_dir, faces, owner, patches):
    """推导每个 cell 的角点（union of 相邻 face 的顶点）。
    返回 dict: cell -> sorted list of openfoam point/face-vertex indices。"""
    bound_set = set()
    for p in patches:
        bound_set.update(range(p["start"], p["start"] + p["nFaces"]))
    internal = [f for f in range(len(faces)) if f not in bound_set]
    # 读 neighbour（每个内部面的对面 cell）
    ntxt = _read(os.path.join(case_dir, "constant", "polyMesh", "neighbour"))
    _, nbody = _after_int_block(ntxt)
    neigh = [int(x) for x in re.findall(r"-?\d+", nbody)]
    cell_faces = {}
    for f in range(len(faces)):
        c = owner[f] if f < len(owner) else -1
        if c >= 0:
            cell_faces.setdefault(c, set()).add(f)
    for i, f in enumerate(internal):
        if i < len(neigh) and neigh[i] >= 0:
            cell_faces.setdefault(neigh[i], set()).add(f)
    corners = {}
    for c, fs in cell_faces.items():
        verts = set()
        for f in fs:
            if f < len(faces):
                verts.update(faces[f])
        if len(verts) == 8:
            corners[c] = sorted(verts)
    return corners


def _write_msh(case_dir, out_dir, faces, points, patches, cellzones):
    """Gmsh 2D/3D ASCII 网格（4.1 格式）。
    - 节点：OpenFOAM points
    - 表面：4 顶点 face → Gmsh type 7（quad），按 patch 分组
    - 体积：cell 8 角点 → Gmsh type 6（hex8），节点顺序为几何排序近似
    - 物理组：patch(dim2) + cellZone(dim3)
    体积 hex8 的 8 节点顺序非 Gmsh 标准 CCW，导入 Gmsh/Fluent 前建议校验。"""
    npts = len(points)
    # patch 物理组
    phys = {}
    pn = 0
    for p in patches:
        phys[p["name"]] = pn
        pn += 1
    # cellZone 物理组（3D）
    zn = 0
    zone_phys = {}
    for z in cellzones:
        zone_phys[z["name"]] = 100 + zn
        zn += 1

    # 表面 face（4 顶点 → quad）
    bound_set = set()
    patch_of_face = {}
    for p in patches:
        for i in range(p["start"], p["start"] + p["nFaces"]):
            bound_set.add(i)
            patch_of_face[i] = p["name"]

    # cell 角点
    corners = _cell_corner_nodes(case_dir, faces,
                                 _parse_owner(case_dir), patches)

    # PhysicalNames
    phys_names = []
    for p in patches:
        phys_names.append(f'2 "{p["name"]}" {phys[p["name"]]}')
    for z in cellzones:
        phys_names.append(f'3 "{z["name"]}" {zone_phys[z["name"]]}')

    msh = []
    msh.append("$MeshFormat")
    msh.append("4.1 0 8")
    msh.append("$EndMeshFormat")
    msh.append("$PhysicalNames")
    msh.append(str(len(phys_names)))
    msh.extend(phys_names)
    msh.append("$EndPhysicalNames")
    msh.append("$Nodes")
    msh.append(str(npts))
    for i, c in enumerate(points):
        msh.append(f"{i+1} {c[0]:.8e} {c[1]:.8e} {c[2]:.8e}")
    msh.append("$EndNodes")

    elems = []
    eid = 1
    # 表面面片（4 顶点 → type 7）
    for fi in sorted(bound_set):
        v = faces[fi]
        if len(v) != 4:
            continue
        pnum = phys[patch_of_face[fi]]
        elems.append(f"{eid} {pnum} 7 0 4 {v[0]+1} {v[1]+1} {v[2]+1} {v[3]+1}")
        eid += 1
    # 体积 hex8（type 6，节点顺序几何近似）
    for c, verts in corners.items():
        if len(verts) != 8:
            continue
        pnum = None
        for z in cellzones:
            if c in set(z["labels"]):
                pnum = zone_phys[z["name"]]
                break
        pnum = pnum if pnum is not None else 999
        elems.append(f"{eid} {pnum} 6 0 8 " + " ".join(str(x+1) for x in verts))
        eid += 1

    msh.append("$Elements")
    msh.append(str(len(elems)))
    msh.extend(elems)
    msh.append("$EndElements")
    # 头部注释
    msh.insert(0, "$ ChatCFD 导出：mesh.msh (Gmsh 4.1)。表面 quad 为 polyMesh boundary 直接生成；"
                  "体积 hex8 节点顺序为几何排序近似，导入 Gmsh/Fluent 前请用工具校验节点顺序，"
                  "或直接使用 polyMesh_snapshot 原生网格。")
    msh_path = os.path.join(out_dir, "mesh.msh")
    open(msh_path, "w", encoding="utf-8").write("\n".join(msh) + "\n")
    return len(elems)


def _write_report(case_dir, out_dir, quality, patches, cellzones,
                  points, faces, file_manifest):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    case_name = os.path.basename(os.path.normpath(case_dir))
    report = {
        "case": case_name,
        "generated_at": now,
        "mesh_scale": {
            "cells": quality.get("cells", len(faces) and 0),
            "n_cells": quality.get("cells"),
            "n_points": quality.get("points", len(points)),
            "n_faces": quality.get("faces", len(faces)),
            "n_internal_faces": quality.get("internal_faces"),
            "total_volume_m3": quality.get("total_volume"),
        },
        "mesh_quality": {
            "non_orthogonality_max": quality.get("non_ortho_max"),
            "non_orthogonality_avg": quality.get("non_ortho_avg"),
            "skewness_max": quality.get("skewness_max"),
            "aspect_ratio_max": quality.get("aspect_max"),
            "mesh_ok": quality.get("mesh_ok"),
            "mesh_type": "structured hexahedral (blockMesh)" if quality.get("mesh_ok")
                         else "polyMesh",
        },
        "boundary_patches": [
            {"name": p["name"], "type": p["type"], "nFaces": p["nFaces"]}
            for p in patches
        ],
        "cell_zones": [
            {"name": z["name"], "cells": z["count"]} for z in cellzones
        ],
        "files": file_manifest,
        "download": {
            "how": "GET /api/download/{case_id} → mesh.tar.gz（含快照+报告）",
            "formats": {
                "polyMesh_snapshot/": "OpenFOAM 原生网格（最可靠，可复用于再算）",
                "mesh.obj": "外表面三角化，任意 3D 软件查看",
                "mesh.msh": "Gmsh 格式，表面可靠；体积 hex8 节点顺序需校验",
                "mesh_report.json": "网格统计/质量/文件清单",
            },
        },
    }
    rp = os.path.join(out_dir, "mesh_report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return report


def export_mesh(case_dir):
    """生成网格后自动保存。返回 manifest（相对路径 → 字节数，_report 为 dict）。
    每个导出动作独立、失败降级，保证至少快照成功。"""
    out_dir = os.path.join(case_dir, "mesh_export")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}

    points = _parse_points(case_dir)
    faces = _parse_faces(case_dir)
    patches = _parse_boundary(case_dir)
    cellzones = _parse_cellzones(case_dir)
    quality = _mesh_quality(case_dir)

    # 1) 原生快照（最可靠，优先保证）
    try:
        _write_snapshot(case_dir, out_dir)
        manifest["_snapshot"] = "polyMesh_snapshot/"
    except Exception as e:
        manifest["_snapshot_error"] = str(e)

    # 2) .obj 外表面
    try:
        nf = _write_obj(case_dir, out_dir, faces, points, patches)
        manifest["mesh_export/mesh.obj"] = os.path.getsize(os.path.join(out_dir, "mesh.obj"))
        manifest["_obj_triangles"] = nf
    except Exception as e:
        manifest["_obj_error"] = str(e)

    # 3) .msh Gmsh
    try:
        elems = _write_msh(case_dir, out_dir, faces, points, patches, cellzones)
        manifest["mesh_export/mesh.msh"] = os.path.getsize(os.path.join(out_dir, "mesh.msh"))
        manifest["_msh_elements"] = elems
    except Exception as e:
        manifest["_msh_error"] = str(e)

    # 4) 报告 + 5) 打包
    file_manifest = dict(manifest)
    try:
        report = _write_report(case_dir, out_dir, quality, patches, cellzones,
                               points, faces, file_manifest)
        manifest["_report"] = report
    except Exception as e:
        manifest["_report_error"] = str(e)
    try:
        tgz = os.path.join(out_dir, "mesh.tar.gz")
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(os.path.join(out_dir, "polyMesh_snapshot"),
                    arcname="polyMesh_snapshot")
            for f in ("mesh.obj", "mesh.msh", "mesh_report.json"):
                fp = os.path.join(out_dir, f)
                if os.path.exists(fp):
                    tar.add(fp, arcname=f)
        manifest["mesh_export/mesh.tar.gz"] = os.path.getsize(tgz)
    except Exception as e:
        manifest["_tar_error"] = str(e)

    return manifest
