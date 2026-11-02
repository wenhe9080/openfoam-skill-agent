#!/usr/bin/env python3
"""回归测试：MRF+叶片力源等效模型强制稳态 + 瞬态时间预算收缩

背景（历史踩坑）：
  MRF + fvOptions 叶片力源是「定常等效模型」，物理上没有非定常结构。
  若 LLM 把 controlDict 写成 pimpleFoam 瞬态（endTime=0.5 / deltaT=5e-5），
  需要 10000 步；单核 240s 只能跑约 1863 步（0.04s）→ 必然 timeout
  → 判失败 → LLM 纠错重写 → 再 timeout → 循环 6 轮空烧算力。

本测试验证两个修复函数的行为：
  A) force_steady_for_actuator_case：三条件齐备时 pimpleFoam → simpleFoam
  B) 幂等：已是 simpleFoam 不再改
  C) patch_transient_time_budget：纯瞬态算例 endTime 收缩到预算内
  D) 稳态求解器不受时间预算影响
  E) 无 fvOptions 的普通算例不被强制稳态

用法（在 WSL 内，因 app.py 硬编码 /mnt/g 路径）：
    cd /mnt/g/ChatCFD-main && ./chatcfd_venv/bin/python tests/test_transient_fix.py
"""
import os
import re
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "foam_agent"))
import app as A  # noqa: E402


def _make_case(root, application="pimpleFoam", with_mrf=True, with_fvoptions=True):
    """构造最小算例骨架"""
    os.makedirs(os.path.join(root, "system"), exist_ok=True)
    os.makedirs(os.path.join(root, "constant"), exist_ok=True)
    with open(os.path.join(root, "system", "controlDict"), "w") as f:
        f.write(f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {application};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          5e-05;
writeControl    adjustableRunTime;
writeInterval   0.1;
adjustTimeStep  yes;
maxCo           0.5;
""")
    if with_mrf:
        with open(os.path.join(root, "constant", "MRFProperties"), "w") as f:
            f.write("""FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }
MRF1
{
    cellZone            rotor;
    active              yes;
    nonRotatingPatches  (casing);
    origin              (0 0 0);
    axis                (1 0 0);
    omega               157.08;
}
""")
    if with_fvoptions:
        with open(os.path.join(root, "constant", "fvOptions"), "w") as f:
            f.write("""FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }
rotorForce
{
    type            vectorSemiImplicitSource;
    active          yes;
    selectionMode   cellZone;
    cellZone        rotor;
    volumeMode      specific;
    sources
    {
        U               ((0 0 3356.21) 0);
    }
}
""")
    # checkMesh 日志（用于估算 cell 数）
    with open(os.path.join(root, "log.checkMesh"), "w") as f:
        f.write("    cells:            9600\n")
    # 求解日志（提供 deltaT 历史）
    with open(os.path.join(root, "log.solver"), "w") as f:
        f.write("Time = 0.00005\ndeltaT = 5e-05\nTime = 0.0001\ndeltaT = 5e-05\n")
    return root


def _cd_vals(case):
    txt = open(os.path.join(case, "system", "controlDict"), encoding="utf-8").read()
    def get(k):
        m = re.search(rf"^{k}\s+([0-9.eE+-]+|\w+)", txt, re.M)
        return m.group(1) if m else None
    return get("application"), get("endTime"), get("deltaT"), get("writeInterval")


def main():
    results = []
    tmp_root = tempfile.mkdtemp(prefix="chatcfd_transient_fix_")

    # A) 强制稳态
    case = _make_case(os.path.join(tmp_root, "a"))
    before = _cd_vals(case)
    changed, note = A.force_steady_for_actuator_case(case)
    after = _cd_vals(case)
    ok = changed and after[0] == "simpleFoam" and float(after[1]) == 1000 and float(after[2]) == 1
    results.append(("A 强制稳态 pimpleFoam→simpleFoam", ok,
                    f"before={before[:3]} after={after[:3]}"))

    # B) 幂等
    changed2, _ = A.force_steady_for_actuator_case(case)
    results.append(("B 幂等（已稳态不再改）", not changed2, f"changed={changed2}"))

    # C) 瞬态时间预算收缩
    case = _make_case(os.path.join(tmp_root, "c"))
    before = _cd_vals(case)
    ch, note = A.patch_transient_time_budget(case, "pimpleFoam", 240)
    after = _cd_vals(case)
    ok = ch and float(after[1]) < float(before[1]) and float(after[1]) > 0
    results.append(("C 瞬态 endTime 收缩", ok,
                    f"endTime {before[1]} → {after[1]}"))

    # D) 稳态不受影响
    case = _make_case(os.path.join(tmp_root, "d"), application="simpleFoam")
    ch, _ = A.patch_transient_time_budget(case, "simpleFoam", 240)
    results.append(("D 稳态不受时间预算影响", not ch, f"changed={ch}"))

    # E) 无 fvOptions 不强制稳态
    case = _make_case(os.path.join(tmp_root, "e"), with_fvoptions=False)
    ch, _ = A.force_steady_for_actuator_case(case)
    results.append(("E 无 fvOptions 不强制稳态", not ch, f"changed={ch}"))

    print("=" * 72)
    for name, ok, info in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {info}")
    print("=" * 72)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"{n_pass}/{len(results)} PASS")

    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
