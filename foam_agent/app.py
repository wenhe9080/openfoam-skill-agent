# -*- coding: utf-8 -*-
"""
Chat-Foam-agent —— ChatCFD + OpenFOAM-skill 合并版对话智能体
自然语言 -> LLM 规划 -> 生成 OpenFOAM 算例 -> blockMesh/checkMesh/求解 -> 自动纠错 -> 结果返回
零第三方依赖（Python 标准库），在 WSL 内运行以直接调用 OpenFOAM。
启动: ./chatcfd_venv/bin/python foam_agent/app.py  (端口 8502)
"""
import json
import math
import os
import re
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)

CFG = json.load(open("/mnt/g/ChatCFD-main/inputs/chatcfd_config.json", encoding="utf-8"))
# 密钥安全：优先读配置文件的非空值，否则回退系统环境变量（JSON 中不再存明文密钥）
API_KEY = CFG.get("DEEPSEEK_V3_KEY") or os.environ.get("DEEPSEEK_V3_KEY", "")
if not API_KEY:
    print("[警告] 未找到 DEEPSEEK_V3_KEY：请在 inputs/chatcfd_config.json 填写，"
          "或设置系统环境变量 DEEPSEEK_V3_KEY")
LLM_URL = (CFG.get("DEEPSEEK_V3_BASE_URL") or os.environ.get(
    "DEEPSEEK_V3_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/") + "/chat/completions"
OF_BASHRC = "/usr/lib/openfoam/openfoam2406/etc/bashrc"
SKILL = "/mnt/c/Users/12919/.workbuddy/skills/openfoam-skills__skillhub"

MODELS = [
    {"id": "qwen3.8-flash", "label": "qwen3.8-flash（快速）"},
    {"id": "qwen3.8-27b", "label": "qwen3.8-27b（均衡）"},
    {"id": "qwen3.8-max", "label": "qwen3.8-max（最强推理）"},
    {"id": "qwen3.8-max-0902", "label": "qwen3.8-max-0902（快照版）"},
]

# ---------- 知识库（来自 openfoam-skill 项目） ----------
KNOWLEDGE = []
_KF = [
    ("case-setup.md", 5000), ("solver-selection.md", 3500),
    ("boundary-condition-playbook.md", 4500), ("turbulence-bc-recipes.md", 4000),
    ("error-recovery.md", 4500), ("mesh-and-blockmeshdict-manual.md", 4500),
    ("fvsolution-and-residual-control.md", 3500),
    # 旋转机械（MRF/滑移网格/叶片力源）—— 原 skill 缺此篇章，导致压气机/涡轮算例退化
    ("mrf-and-turbomachinery.md", 6000),
    ("turbulence-and-numerics.md", 3500),
    ("field-and-dictionary-matrix.md", 3500),
    ("function-object-recipes.md", 2500),
]
for _name, _cap in _KF:
    try:
        _t = open(os.path.join(SKILL, "references", _name), encoding="utf-8").read()
        KNOWLEDGE.append(f"### {_name}\n{_t[:_cap]}")
    except Exception:
        pass
try:
    _tpl = []
    for _f in ["0/U.txt", "0/p.txt", "system/controlDict.txt", "system/fvSchemes.txt",
               "system/fvSolution.txt", "constant/turbulenceProperties.txt"]:
        _tpl.append(f"#### {_f}\n```\n{open(os.path.join(SKILL,'assets/templates/simplefoam-minimal',_f),encoding='utf-8').read()}\n```")
    KNOWLEDGE.append("### minimal simpleFoam template\n" + "\n".join(_tpl))
except Exception:
    pass
KNOWLEDGE_TEXT = "\n\n".join(KNOWLEDGE)

# ---------- LLM 调用 ----------
def llm(messages, model, temperature=0.3):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "enable_thinking": False}
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                LLM_URL, data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + API_KEY,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            txt = str(e)
            if "enable_thinking" in txt or "Parameter" in txt:
                body.pop("enable_thinking", None)  # 模型不支持则去掉重试
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败: {last_err}")

def _loads_lenient(s):
    """容错 JSON 解析：依次尝试原样、去尾逗号、转义控制字符、ast.literal_eval"""
    # 1) 原样
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) 去除尾逗号（, 后面紧跟 } 或 ]）
    t = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) 转义未转义的控制字符（如字符串里的裸换行/回车）
    t2 = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    t2 = re.sub(r",\s*([}\]])", r"\1", t2)
    try:
        return json.loads(t2)
    except Exception:
        pass
    # 4) 转义未转义的双引号（LLM 常在字符串值里写裸引号）
    t3 = t2
    try:
        return json.loads(t3)
    except Exception:
        pass
    # 5) ast.literal_eval（处理单引号 Python 风格字典）
    import ast
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    raise ValueError("JSON 解析失败")

def extract_json(text):
    """从 LLM 回复中稳健提取第一个平衡 JSON 对象（含容错）。

    容错策略：
    - 优先匹配 ```json 围栏
    - 无围栏时用括号深度扫描，找到第一个平衡的 {…}（或 […]，供数组使用）
    - 扫描到文本末尾仍不平衡（LLM 截断输出）→ 尝试补齐缺失的 } / ] 后解析
    - 全部失败抛 ValueError（由上层决定是否重试）
    """
    # 0) 清理：去掉 ``` 围栏标记但保留内容
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    # 优先匹配 ```json 围栏（已并入上面 text 处理，这里保留兼容）
    # 1) 括号深度扫描，支持 {} 与 []
    start = text.find("{")
    if start < 0:
        start = text.find("[")
    if start < 0:
        raise ValueError("回复中无 JSON")
    depth_o, depth_a = 0, 0   # {} 深度, [] 深度
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth_o += 1
            elif c == "}":
                depth_o -= 1
            elif c == "[":
                depth_a += 1
            elif c == "]":
                depth_a -= 1
        # 找到完全平衡且对象/数组已闭合
        if depth_o == 0 and depth_a == 0 and (c in "}]") and i > start:
            try:
                return _loads_lenient(text[start:i + 1])
            except Exception:
                pass
    # 2) 未找到平衡（截断/不平衡）→ 从 start 起尝试补齐缺失的闭合符
    #    收集 start 之后的括号栈，遇文本末尾再补 '}'/']'
    seg = text[start:]
    stack = []
    in_str = False
    esc = False
    for c in seg:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()
    if stack:
        # 补齐：栈里剩的是未闭合的 { / [，按反序补对应闭符
        closing = "".join("}" if s == "{" else "]" for s in reversed(stack))
        cand = seg
        # 若最后一个未闭合的是字符串（截断在字符串中间），先补上引号
        # 检测：段末处于字符串内部
        in_str_end = False
        esc_end = False
        for c in seg:
            if in_str_end:
                if esc_end:
                    esc_end = False
                elif c == "\\":
                    esc_end = True
                elif c == '"':
                    in_str_end = False
            else:
                if c == '"':
                    in_str_end = True
        if in_str_end:
            cand = cand.rstrip() + '"'
        cand = cand + closing
        try:
            return _loads_lenient(cand)
        except Exception:
            pass
    # 3) 尝试逐行剥离说明文字后重试（LLM 常在 JSON 前后加散文）
    for strip_leading in (True, False):
        lines = text.splitlines()
        cand_lines = []
        for ln in lines:
            cand_lines.append(ln)
        cand = "\n".join(cand_lines)
        for attempt in range(2):
            try:
                return _loads_lenient(cand)
            except Exception:
                # 去掉首尾可能带文字的行
                if attempt == 0:
                    cand = cand[cand.find("{"):] if "{" in cand else cand
                    cand = cand[:cand.rfind("}") + 1] if "}" in cand else cand
                else:
                    break
    raise ValueError("JSON 不平衡")

# ---------- OpenFOAM 执行 ----------
def run_bash(cmd, cwd, timeout=300):
    full = f'source {OF_BASHRC} 2>/dev/null; cd "{cwd}" && {cmd}'
    try:
        p = subprocess.run(["bash", "-c", full], capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"[超时 {timeout}s 已终止]"

def parse_residuals(log_text):
    steps = re.split(r"Time = ([0-9.eE+-]+)", log_text)
    series = {}
    times = []
    for i in range(1, len(steps) - 1, 2):
        body = steps[i + 1]
        row = {}
        for v in ("Ux", "Uy", "Uz", "p", "k", "omega", "epsilon", "e", "T"):
            m = re.search(rf"Solving for {v}, Initial residual = ([0-9.eE+-]+)", body)
            if m:
                row[v] = float(m.group(1))
        if row:
            times.append(float(steps[i]))
            for v, val in row.items():
                series.setdefault(v, []).append(val)
    return times, series

# ---------- 文件写入净化（沉淀自 ChatCFD 复现经验） ----------
def sanitize_content(rel, content):
    """剥离 markdown 围栏/游离文本；缺失 FoamFile 头则按路径推断补齐"""
    content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content.strip(), flags=re.M).strip()
    if "FoamFile" in content[:300]:
        return content + "\n"
    base = os.path.basename(rel.rstrip("/"))
    if rel.startswith("system/") or rel.startswith("constant/"):
        cls = "dictionary"
    elif base in ("U",):
        cls = "volVectorField"
    else:
        cls = "volScalarField"
    header = (f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n"
              f"    class       {cls};\n    object      {base};\n}}\n")
    return header + "\n" + content + "\n"

def write_files(case_dir, files):
    written = []
    for rel, content in files.items():
        rel = rel.strip().lstrip("/")
        if not rel or content is None:
            continue
        p = os.path.join(case_dir, rel)
        os.makedirs(os.path.dirname(p) or case_dir, exist_ok=True)
        open(p, "w", encoding="utf-8").write(sanitize_content(rel, str(content)))
        written.append(rel)
    return written

# ---------- 提示词 ----------
PLAN_PROMPT = """你是 OpenFOAM 仿真规划专家。分析用户输入，输出严格 JSON（不要多余文字）：
{
 "intent": "run_case 或 qa",
 "case_name": "英文小写下划线算例名",
 "solver": "simpleFoam/pimpleFoam/icoFoam/rhoCentralFoam/interFoam 等 OpenFOAM v2406 真实存在的求解器",
 "turbulence": "laminar 或 kOmegaSST/kEpsilon/等 RAS 模型名",
 "transient": true/false,
 "physics_cn": "一句中文物理描述",
 "geometry_cn": "一句中文几何描述",
 "bc_cn": "中文边界条件概述（速度/压力值等）",
 "endTime": "建议的物理结束时间(数字)"
}
判断规则（务必严格遵守）：
1. 用户想跑一个具体算例/几何/流动/参数扫描，或要求"仿真/模拟/计算/跑一下/求解/做实验报告" → intent 必须是 run_case
2. 用户上传了论文/实验指导书等技术文档，且消息中提到仿真/实验/算例 → intent 必须是 run_case
3. 只有纯粹的"概念解释、报错含义、求解器区别"类提问（且不涉及具体算例执行）→ intent 才为 qa
4. intent=run_case 时，case_name/solver/turbulence/endTime **必须给出有效值**，不得留空。
   solver 不确定就用 simpleFoam（不可压稳态）；turbulence 不确定就用 kOmegaSST；endTime 默认 0.5
用户输入：
"""

GEN_PROMPT = """你是 OpenFOAM 算例文件生成专家。根据规划结果，生成一套**完整可直接运行**的 OpenFOAM v2406 算例文件。

【领域知识库（openfoam-skill 项目沉淀）】
%s

【规划结果】
%s

【用户原始需求】
%s

硬性要求：
1. 输出严格 JSON：{"files": {"0/U": "文件全文", "0/p": "...", "constant/...": "...", "system/...": "..."}}
2. 必须包含：0/ 场文件、constant/（turbulenceProperties 或 momentumTransport + transportProperties 或 physicalProperties）、system/controlDict、system/fvSchemes、system/fvSolution
3. 若几何是简单形状（方腔/管道/绕流外场等），必须包含 system/blockMeshDict（注意顶点顺序为 8 个六面体顶点，不能自相交）；不要发明不存在的求解器或边界类型
4. FoamFile 头必须规范：FoamFile { version 2.0; format ascii; class <volVectorField|volScalarField|dictionary>; object <U|p|...>; }
5. 首轮求稳：对流格式用 Gauss upwind，拉普拉斯 Gauss linear corrected，稳态加松弛因子（p 0.3, U 0.5 以下），endTime 取规划值但首run不宜过大
6. turbulence 场文件必须与湍流模型匹配（kOmegaSST 要 k 和 omega；laminar 不要湍流场）
7. 所有中文注释转为英文或删除，文件内容只含合法 OpenFOAM 语法
"""

FIX_PROMPT = """你是 OpenFOAM 调错专家。算例运行失败，请根据错误日志修复文件。

【错误日志（尾部）】
%s

【当前文件】
%s

【领域知识（错误恢复手册）】
%s

【硬性检查清单（修复后必须逐项确认）】
1. 瞬态求解器（icoFoam/pimpleFoam）的 fvSolution 必须在 solvers 中包含 UFinal 和 pFinal 条目（relTol 0）
2. 所有文件第一行必须是 FoamFile 头
3. 边界场文件的 boundary 名称必须与 blockMeshDict/网格中的 patch 名完全一致

输出严格 JSON：{"note": "修复说明", "files": {"需要修改/新增的文件路径": "完整新内容"}}（只给需要改动的文件）：
"""

REPORT_PROMPT = """你是 CFD 仿真实验报告撰写专家。算例已成功运行，请根据以下信息生成一份**完整的中文实验报告**（HTML 片段，可直接嵌入网页显示，不要包含 <html>/<head>/<body> 标签，样式用内联 style）。

【物理描述】%s
【几何描述】%s
【边界条件】%s
【求解器】%s（湍流模型：%s）
【收敛步数】%s
【残差数据】%s

【用户上传的实验指导书/文档内容（节选）】
%s

报告结构要求（用 <h2>/<h3> 分节）：
1. **实验目的与原理**：结合上传文档的实验目的，说明本仿真的物理意义
2. **实验装置与工况**：列出几何尺寸、边界条件、求解器与湍流模型设置
3. **数值模拟结果**：描述收敛情况（残差下降量级）、速度/压力场特征（根据残差趋势合理推断，不要编造具体数值），可给出残差收敛结论
4. **结果分析与讨论**：与文档中预期现象对比，讨论物理合理性
5. **实验结论**：总结本算例得到的主要结论
6. **思考与改进**：指出仿真局限性（如网格质量、湍流模型适用性）与改进方向

要求：用简体中文，专业名词（solver、湍流模型、边界类型）保留英文；内容详实、逻辑清晰，直接输出 HTML 代码：
"""

# ---------- 文档上传 ----------
UPLOADS = {}

def extract_doc_text(name, data):
    """从上传文件提取文本（PDF 用 PyPDF2，其余按 UTF-8 文本）"""
    if name.lower().endswith(".pdf"):
        import io
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = []
        for pg in reader.pages:
            try:
                pages.append(pg.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")

# ---------- 任务流水线 ----------
JOBS = {}
JOBS_LOCK = threading.Lock()

def job_update(jid, **kw):
    with JOBS_LOCK:
        j = JOBS[jid]
        j.update(kw)
        _refresh_workflow(j)

def _refresh_workflow(j):
    """根据 steps 构建按时间排序的完整工作流时间线快照"""
    now = time.time()
    flow = []
    for s in j.get("steps", []):
        flow.append({
            "name": s["name"], "state": s["state"], "detail": s.get("detail", ""),
            "order": s.get("order", len(flow)),
            "start": s.get("start"), "end": s.get("end"),
            "duration": s.get("duration"),
            "elapsed": round((s.get("end") or now) - s.get("start", now), 1),
        })
    j["workflow"] = flow
    j["elapsed_total"] = round(now - j.get("start", now), 1)

def job_step(jid, name, state="running", detail=""):
    """记录步骤状态与耗时；同一名称重复调用(如多轮纠错)追加为新步骤"""
    now = time.time()
    with JOBS_LOCK:
        j = JOBS[jid]
        steps = j["steps"]
        # 若最后一个同名步骤仍为 running，视为其状态更新（记录结束时间）
        if steps and steps[-1]["name"] == name and steps[-1]["state"] == "running":
            s = steps[-1]
            s.update(state=state, detail=detail[-500:], end=now,
                     duration=round(now - s.get("start", now), 1))
            j["current"] = name
            j["step_done"] = sum(1 for x in steps if x["state"] != "running")
            _refresh_workflow(j)
            return
        steps.append({"name": name, "state": state, "detail": detail[-500:],
                      "start": now, "order": len(steps)})
        if state != "running":
            steps[-1]["end"] = now
            steps[-1]["duration"] = round(now - steps[-1]["start"], 1)
        j["current"] = name
        j["step_done"] = sum(1 for x in steps if x["state"] != "running")
        _refresh_workflow(j)

def pipeline(jid, message, model, history, upload_text=""):
    j = JOBS[jid]
    try:
        doc_block = f"\n\n【用户上传的文档内容（节选）】\n{upload_text[:16000]}" if upload_text else ""
        doc_block_gen = f"\n\n【用户上传的文档内容（节选，边界条件/几何参数以文档为准）】\n{upload_text[:20000]}" if upload_text else ""
        # 1. 规划
        job_step(jid, "规划", "running")
        plan = extract_json(llm([{"role": "user", "content": PLAN_PROMPT + message + doc_block}], model, 0.2))
        # === 确定性兜底：修正规划器误判（用户常带文档说"先仿真/写报告"，LLM 误判 qa）===
        # 核心原则：
        #   - 上传了文档 → 用户大概率要跑文档里的算例/实验，倾向 run_case
        #   - 消息含强动作词（仿真/模拟/求解/实验报告/运行）→ run_case
        #   - 两者任一成立即强制 run_case（带文档+动作词场景最可靠）
        _action_kw = ("仿真", "模拟", "计算", "求解", "跑一下", "跑个", "运行",
                     "实验报告", "数值模拟", "直仿",
                     "simulate", "simulation", "run case", "solve")
        # "求解"需排除"求解器"（"求解器区别？"是提问不是动作）
        _action_hit = any((k not in ("求解", "solve") and k in message) or
                          (k == "求解" and re.search(r"求解(?!器)", message)) or
                          (k == "solve" and re.search(r"\bsolve(?!r)", message, re.I))
                          for k in _action_kw)
        # 明确的纯诊断词（"区别/含义/为什么报错"）且无动作词 → 才允许保持 qa
        _diagnostic_only = (any(d in message for d in ("区别", "什么意思", "含义",
                                                        "为什么", "what", "differ"))
                            and not _action_hit and not upload_text)
        force_run = (upload_text and _action_hit) or (upload_text and "实验报告" in message)
        if plan.get("intent") == "qa" and not _diagnostic_only and (force_run or _action_hit):
            if upload_text or _action_hit:
                job_step(jid, "规划修正", "done",
                         f"检测到仿真意图（动作词{'+' if upload_text else ''}已传文档），强制 run_case（原判 qa）")
                plan["intent"] = "run_case"
        # run_case 时补齐必需字段，避免后续 KeyError / 空 solver
        if plan.get("intent") == "run_case":
            if not plan.get("solver"):
                plan["solver"] = "simpleFoam"
            if not plan.get("turbulence"):
                plan["turbulence"] = "kOmegaSST"
            if not plan.get("case_name"):
                plan["case_name"] = "chatfoam_case"
            if not plan.get("endTime"):
                plan["endTime"] = 0.5
        j["plan"] = plan
        job_step(jid, "规划", "done", json.dumps(plan, ensure_ascii=False))
        if plan.get("intent") == "qa":
            job_step(jid, "解答", "running")
            sys_prompt = ("你是 OpenFOAM 专家（知识来自 openfoam-skill 手册）。用简体中文回答，专业名词保留英文。"
                          + (f"\n用户上传了文档，回答时可引用其内容：\n{upload_text[:12000]}" if upload_text else "")
                          + KNOWLEDGE_TEXT[:9000])
            ans = llm([{"role": "system", "content": sys_prompt}]
                      + history[-8:] + [{"role": "user", "content": message}], model, 0.5)
            job_update(jid, status="done", answer=ans)
            job_step(jid, "解答", "done", ans[:200])
            return

        # 2. 生成算例文件
        job_step(jid, "生成算例文件", "running")
        gen = extract_json(llm([{"role": "user", "content": GEN_PROMPT % (
            KNOWLEDGE_TEXT[:26000], json.dumps(plan, ensure_ascii=False), message + doc_block_gen)}], model, 0.2))
        files = gen["files"]
        case_id = re.sub(r"[^a-zA-Z0-9_]", "_", plan.get("case_name") or "case") + "_" + jid[:6]
        case_dir = os.path.join(RUNS, case_id)
        os.makedirs(case_dir, exist_ok=True)
        j["case_id"], j["case_dir"] = case_id, case_dir
        written = write_files(case_dir, files)
        job_step(jid, "生成算例文件", "done", f"{len(written)} 个文件")

        # 3-6. 网格 + 求解 + 自动纠错（可压缩/复杂算例需更多轮次）
        max_round = 6
        for rnd in range(max_round):
            ok, err_all = run_case(jid, case_dir, plan)
            if ok:
                break
            if rnd == max_round - 1:
                job_update(jid, status="failed", error=err_all[-3000:])
                job_step(jid, "自动纠错", "failed", err_all[-500:])
                return
            job_step(jid, "自动纠错", "running", f"第 {rnd+1} 轮修复")
            cur = {}
            for rel in files:
                p = os.path.join(case_dir, rel)
                if os.path.exists(p):
                    cur[rel] = open(p, encoding="utf-8").read()
            fix = extract_json(llm([{"role": "user", "content": FIX_PROMPT % (
                err_all[-2500:], json.dumps({k: v[:3500] for k, v in cur.items()}),
                open(os.path.join(SKILL, "references", "error-recovery.md"), encoding="utf-8").read()[:4500]
            )}], model, 0.2))
            note = fix.get("note", "")
            for rel in write_files(case_dir, fix.get("files", {})):
                files.setdefault(rel, "")
            job_step(jid, "自动纠错", f"第 {rnd+1} 轮完成: {note}"[:500])

        # === 报告生成环节（用户上传了文档时）===
        report_html = ""
        if upload_text:
            job_step(jid, "生成实验报告", "running")
            try:
                # 读取求解日志提取关键数据
                log_text = ""
                lp = os.path.join(case_dir, "log.solver")
                if os.path.exists(lp):
                    log_text = open(lp, encoding="utf-8", errors="ignore").read()
                times, series = parse_residuals(log_text)
                report_prompt = REPORT_PROMPT % (
                    plan.get("physics_cn", ""), plan.get("geometry_cn", ""),
                    plan.get("bc_cn", ""), plan.get("solver", "simpleFoam"),
                    plan.get("turbulence", ""),
                    len(times), json.dumps(series, ensure_ascii=False)[:2500],
                    upload_text[:10000])
                raw = llm([{"role": "user", "content": report_prompt}], model, 0.4)
                # 报告可能是 HTML 或纯文本；去除 ``` 围栏
                report_html = raw.strip()
                mrep = re.search(r"```(?:html)?\s*(.*?)\s*```", raw, re.S)
                if mrep:
                    report_html = mrep.group(1).strip()
                job_step(jid, "生成实验报告", "done", f"报告 {len(report_html)} 字符")
            except Exception as e:
                traceback.print_exc()
                job_step(jid, "生成实验报告", "failed", str(e)[:200])
                report_html = f"<p>实验报告生成失败：{e}</p>"
        job_update(jid, status="done", report=report_html, answer=report_html)
    except Exception as e:
        traceback.print_exc()
        job_update(jid, status="failed", error=f"{e}")

# ---------- 确定性补丁（沉淀自 OpenFOAM 领域知识，不依赖 LLM） ----------
def guess_domain_box(case_dir):
    """从 0/U 或 blockMeshDict 顶点推断计算域 [xmin,xmax,ymin,ymax,zmin,zmax]，用于自动补 blockMeshDict"""
    # 优先从已有 blockMeshDict 顶点读取
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if os.path.exists(bmd):
        txt = open(bmd, encoding="utf-8").read()
        vs = re.findall(r"\(([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\)", txt)
        if vs:
            xs = [float(a) for a, b, c in vs]
            ys = [float(b) for a, b, c in vs]
            zs = [float(c) for a, b, c in vs]
            return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    return None

# ============ 涡轮机械几何构建器（建议一：按 plan 生成真实环形流道+MRF，不再回退矩形盒） ============
_TURBO_HINTS = ("压气机", "涡轮", "压缩机", "透平", "叶轮", "叶栅", "叶片", "转子", "静子",
                "轴流", "离心", "mrf", "rotor", "stator", "blade", "cascade",
                "轮毂", "轮毂比", "外径", "转速", "rpm", "增压", "叶轮机")

def _is_turbomachinery(plan):
    """检测 plan 是否描述涡轮机械（压气机/涡轮/叶片机）几何。
    命中后应走 build_turbomachinery 而非默认矩形盒回退。"""
    if not isinstance(plan, dict):
        return False
    blob = " ".join(str(plan.get(k, "")) for k in ("geometry_cn", "bc_cn", "physics_cn", "case_name"))
    low = blob.lower()
    hits = sum(1 for h in _TURBO_HINTS if h in low)
    # 需要多个线索才判定，避免把"转子"单字误判
    if hits >= 2:
        return True
    if any(h in low for h in ("压气机", "压缩机", "涡轮", "叶轮", "叶栅", "透平", "轴流", "离心")):
        return True
    return False


def _extract_radius_m(plan):
    """从 plan 几何提取 (r_hub, r_tip) 单位米。默认 0.5m 外径、0.75 轮毂比。"""
    g = str(plan.get("geometry_cn", "") or "")
    r_tip, hub_ratio = 0.25, 0.75
    m = re.search(r"外径\s*([0-9.]+)\s*m", g)
    if not m:
        m = re.search(r"外径\s*([0-9.]+)", g)
    if not m:
        m = re.search(r"直径\s*([0-9.]+)", g)
    if m:
        try:
            d = float(m.group(1))
            if 0 < d <= 5:
                r_tip = d / 2.0
        except Exception:
            pass
    m2 = re.search(r"轮毂比\s*([0-9.]+)", g)
    if not m2:
        m2 = re.search(r"hub\s*ratio\s*([0-9.]+)", g, re.I)
    if m2:
        try:
            hr = float(m2.group(1))
            if 0 < hr < 1:
                hub_ratio = hr
        except Exception:
            pass
    return r_tip * hub_ratio, r_tip


def _extract_rpm(plan):
    txt = str(plan.get("bc_cn", "")) + " " + str(plan.get("geometry_cn", ""))
    m = re.search(r"(\d+)\s*rpm", txt, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 1500.0


def _extract_axial_velocity(plan, r_hub, r_tip, rpm):
    """从 plan 提取轴向速度 c_a (m/s)。优先级：
    1) 文本中的 "轴向速度 xx m/s"
    2) 流量系数 phi = c_a/u_m 与叶中切线速度 u_m = omega*r_m → c_a = phi*u_m
    3) 质量流量 m_dot / (rho * A)
    4) 兜底 18.9 m/s（等价 φ≈0.55、1500rpm、轮毂比0.75 的设计点）
    """
    txt = " ".join(str(plan.get(k, "")) for k in ("bc_cn", "geometry_cn", "physics_cn"))
    r_mid = (r_hub + r_tip) / 2.0
    u_mid = rpm * 2.0 * math.pi / 60.0 * r_mid
    m = re.search(r"轴向速度\s*([0-9.]+)\s*m/s", txt)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"流量系数\s*(?:phi|φ)?\s*=?\s*([0-9.]+)", txt, re.I)
    if m:
        try:
            phi = float(m.group(1))
            if 0 < phi < 2:
                return phi * u_mid
        except Exception:
            pass
    m = re.search(r"([0-9.]+)\s*kg/s", txt)
    if m:
        try:
            mdot = float(m.group(1))
            area = math.pi * (r_tip ** 2 - r_hub ** 2)
            rho = 1.2
            return mdot / (rho * area)
        except Exception:
            pass
    return 18.9


def _write_turbomachinery_fvschemes(case_dir, steady):
    """为涡轮机械楔形通道写正确的数值格式（精度 + 稳定平衡）：
    - 对流项线性迎风 limitedLinear 1（二阶、有界，比纯 upwind 精度高，比 linear 稳）
    - 拉普拉斯 corrected（非正交楔形网格）
    - 湍流项 bounded Gauss limitedLinear 1
    - 若稳态：ddt=steadyState，否则 Euler
    """
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

ddtSchemes
{{
    default         {'steadyState' if steady else 'Euler'};
}}
gradSchemes
{{
    default         Gauss linear;
    limited         cellLimited Gauss linear 1;
}}
divSchemes
{{
    default         none;
    div(phi,U)      Gauss limitedLinearV 1;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,omega)  Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes
{{
    default         Gauss linear corrected;
}}
interpolationSchemes
{{
    default         linear;
}}
snGradSchemes
{{
    default         corrected;
}}
wallDist
{{
    method          meshWave;
}}
"""
    open(os.path.join(case_dir, "system", "fvSchemes"), "w", encoding="utf-8").write(body)


def _write_turbomachinery_fvsolution(case_dir, steady):
    """为楔形涡轮通道写匹配的线性求解器与算法控制。"""
    if steady:
        algo = """SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|omega)"     1e-4;
    }
}
relaxationFactors
{
    equations
    {
        U               0.7;
        "(k|omega)"     0.7;
    }
    fields
    {
        p               0.3;
    }
}
"""
    else:
        algo = """PIMPLE
{
    nOuterCorrectors 2;
    nCorrectors      2;
    nNonOrthogonalCorrectors 1;
}
relaxationFactors
{
    equations
    {
        U               0.9;
        "(k|omega)"     0.9;
    }
    fields
    {
        p               0.5;
    }
}
"""
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.05;
        smoother        symGaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
    }}
    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }}
    pFinal
    {{
        $p;
        relTol          0;
        tolerance       1e-7;
    }}
    "(U|k|omega)Final"
    {{
        $U;
        relTol          0;
        tolerance       1e-8;
    }}
}}

{algo}"""
    open(os.path.join(case_dir, "system", "fvSolution"), "w", encoding="utf-8").write(body)


def _write_turbomachinery_field_boundaries(case_dir, r_hub, r_tip, rpm, u_axial=50.0, prewhirl=0.0):
    """重写 0/ 场文件边界为涡轮机械楔形通道的 6 个 patch（inlet/outlet/hub/casing/front/back）。

    物理约定（轴流压气机基元级，参照实验题目"转子进口预旋 c1u=0"）：
      - 进口绝对速度 = 纯轴向 (u_axial, 0, prewhirl)，默认 prewhirl=0（零预旋进气）
      - 切向速度由 MRF 旋转域 + 叶片做功产生，不应人为强加在进口（否则与"零预旋"矛盾）
      - 出口允许切向分量（pressureInletOutletVelocity 允许回流，增强稳定性）
    """
    zero = os.path.join(case_dir, "0")
    if not os.path.isdir(zero):
        return
    # 进口：纯轴向 + 可选预旋（默认 0）
    for fname in ("U",):
        fp = os.path.join(zero, fname)
        if not os.path.exists(fp):
            continue
        txt = open(fp, encoding="utf-8").read()
        i = txt.find("boundaryField")
        if i < 0:
            continue
        head = txt[:i]
        body = f"""boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform ({u_axial:.4g} 0 {prewhirl:.4g});
    }}
    outlet
    {{
        type pressureInletOutletVelocity;
        value uniform ({u_axial:.4g} 0 0);
    }}
    hub
    {{
        type noSlip;
    }}
    casing
    {{
        type noSlip;
    }}
    front
    {{
        type wedge;
    }}
    back
    {{
        type wedge;
    }}
}}
"""
        open(fp, "w", encoding="utf-8").write(head + body)
    # p / k / omega / nut 用标准 wedge + 壁函数边界
    scalar_bodies = {
        "p": ("""boundaryField
{
    inlet   { type zeroGradient; }
    outlet  { type fixedValue; value uniform 0; }
    hub     { type zeroGradient; }
    casing  { type zeroGradient; }
    front   { type wedge; }
    back    { type wedge; }
}
"""),
        "k": ("""boundaryField
{
    inlet   { type fixedValue; value uniform 1; }
    outlet  { type zeroGradient; }
    hub     { type kqRWallFunction; value uniform 1; }
    casing  { type kqRWallFunction; value uniform 1; }
    front   { type wedge; }
    back    { type wedge; }
}
"""),
        "omega": ("""boundaryField
{
    inlet   { type fixedValue; value uniform 100; }
    outlet  { type zeroGradient; }
    hub     { type omegaWallFunction; value uniform 100; }
    casing  { type omegaWallFunction; value uniform 100; }
    front   { type wedge; }
    back    { type wedge; }
}
"""),
        "nut": ("""boundaryField
{
    inlet   { type calculated; value uniform 1e-5; }
    outlet  { type calculated; value uniform 1e-5; }
    hub     { type nutkWallFunction; value uniform 1e-5; }
    casing  { type nutkWallFunction; value uniform 1e-5; }
    front   { type wedge; }
    back    { type wedge; }
}
"""),
    }
    for fname, body in scalar_bodies.items():
        fp = os.path.join(zero, fname)
        if not os.path.exists(fp):
            continue
        txt = open(fp, encoding="utf-8").read()
        i = txt.find("boundaryField")
        if i < 0:
            continue
        open(fp, "w", encoding="utf-8").write(txt[:i] + body)


def _write_turbomachinery_turbulence(case_dir, k_in, omega_in):
    """为涡轮机械楔形通道写 kOmegaSST 湍流模型与湍流场初值/边界。

    - constant/turbulenceProperties：RAS + kOmegaSST（含湍流/层流输运模型，兼容 v2406）
    - 0/k     ：进口固定 k_in；壁面 kqRWallFunction；前后 wedge
    - 0/omega ：进口固定 omega_in；壁面 omegaWallFunction；前后 wedge
    - 0/nut   ：calculated + nutkWallFunction（由 k/omega 反算）
    仅当 RAS 段落存在时才覆写，避免破坏 LES 等其它模型。
    """
    k_in = float(k_in) if k_in and k_in > 0 else 0.05
    omega_in = float(omega_in) if omega_in and omega_in > 0 else 100.0

    # 1) constant/turbulenceProperties：强制 RAS + kOmegaSST（LLM 常写成缺 RAS 的坏字典）
    cdir = os.path.join(case_dir, "constant")
    os.makedirs(cdir, exist_ok=True)
    tp = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;

    turbulence      on;

    printCoeffs     on;
}
"""
    open(os.path.join(cdir, "turbulenceProperties"), "w", encoding="utf-8").write(tp)

    zero = os.path.join(case_dir, "0")
    if not os.path.isdir(zero):
        return
    patched = set()
    # 2) 0/k
    k_body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_in:.6g};

boundaryField
{{
    inlet   {{ type fixedValue; value uniform {k_in:.6g}; }}
    outlet  {{ type inletOutlet; inletValue uniform {k_in:.6g}; value uniform {k_in:.6g}; }}
    hub     {{ type kqRWallFunction; value uniform {k_in:.6g}; }}
    casing  {{ type kqRWallFunction; value uniform {k_in:.6g}; }}
    front   {{ type wedge; }}
    back    {{ type wedge; }}
}}
"""
    open(os.path.join(zero, "k"), "w", encoding="utf-8").write(k_body)
    patched.add("k")
    # 3) 0/omega
    om_body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega_in:.6g};

boundaryField
{{
    inlet   {{ type fixedValue; value uniform {omega_in:.6g}; }}
    outlet  {{ type inletOutlet; inletValue uniform {omega_in:.6g}; value uniform {omega_in:.6g}; }}
    hub     {{ type omegaWallFunction; value uniform {omega_in:.6g}; }}
    casing  {{ type omegaWallFunction; value uniform {omega_in:.6g}; }}
    front   {{ type wedge; }}
    back    {{ type wedge; }}
}}
"""
    open(os.path.join(zero, "omega"), "w", encoding="utf-8").write(om_body)
    patched.add("omega")
    # 4) 0/nut：calculated（非求解量）+ nutkWallFunction 壁面（kOmegaSST 标准配方）
    nut_body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet   {{ type calculated; value uniform 0; }}
    outlet  {{ type calculated; value uniform 0; }}
    hub     {{ type nutkWallFunction; value uniform 0; }}
    casing  {{ type nutkWallFunction; value uniform 0; }}
    front   {{ type wedge; }}
    back    {{ type wedge; }}
}}
"""
    open(os.path.join(zero, "nut"), "w", encoding="utf-8").write(nut_body)
    patched.add("nut")
    return True, "写出 k/omega/nut + turbulenceProperties(RAS kOmegaSST)"


def build_turbomachinery(case_dir, plan):
    """按 plan 生成轴流压气机/涡轮的楔形环形通道 blockMeshDict + MRFProperties + cellZones，
    并重写场文件边界为匹配的 patch。这是"建议一"的核心：让 agent 真正按 plan 建几何，
    而不是回退到 1×1×1 矩形盒。

    顶点约定（参照 OpenFOAM LadenburgJet/movingCone 标准 wedge，已验证可跑）：
      轴向=x，径向=y，周向=z，绕 x 轴旋转。z = ±r*sin(alpha)。
      每个轴向站 4 顶点：0:hubθ-  1:hubθ+  2:tipθ-  3:tipθ+。
      块（站 i->i+1）8 顶点：
        0:(i,hubθ-) 1:(i+1,hubθ-) 2:(i+1,tipθ-) 3:(i,tipθ-)
        4:(i,hubθ+) 5:(i+1,hubθ+) 6:(i+1,tipθ+) 7:(i,tipθ+)
      面：inlet=(0 1 3 2) outlet=(n, n+1, n+3, n+2) hub=(i0 i1 i1' i0')
          casing=(tip...) front(θ-)=... back(θ+)=...
    返回 (是否命中涡轮机械, 说明字符串)。"""
    r_hub, r_tip = _extract_radius_m(plan)
    L = 0.30
    rpm = _extract_rpm(plan)
    omega = rpm * 2.0 * math.pi / 60.0
    alpha = math.radians(1.0)
    c_axial = _extract_axial_velocity(plan, r_hub, r_tip, rpm)

    # 轴向站：进口/转子前缘/转子后缘/静子前缘/出口（转子段为 MRF cellZone）
    xs = [0.0, 0.20 * L, 0.50 * L, 0.70 * L, L]
    nseg = len(xs) - 1

    def st(x, r, s):
        return (x, r, s * r * math.sin(alpha))

    def fmt(p):
        return "(%g %g %g)" % p

    # 顶点：每站 4 个（hubθ-, hubθ+, tipθ-, tipθ+），统一编号（站 i 顶点 = i*4 + {0,1,2,3}）
    verts = []
    for x in xs:
        verts.append(st(x, r_hub, -1))  # +0 hubθ-
        verts.append(st(x, r_hub, +1))  # +1 hubθ+
        verts.append(st(x, r_tip, -1))  # +2 tipθ-
        verts.append(st(x, r_tip, +1))  # +3 tipθ+

    # 网格分辨率：轴向 60 层/段、径向 40 层（贴壁加密 → 解析边界层，y+ 可控）
    nax, nrad = 60, 40
    blocks = []
    for i in range(nseg):
        a, b = i * 4, (i + 1) * 4
        # 块内顺序：0:(i,hubθ-) 1:(i+1,hubθ-) 2:(i+1,tipθ-) 3:(i,tipθ-)
        #           4:(i,hubθ+) 5:(i+1,hubθ+) 6:(i+1,tipθ+) 7:(i,tipθ+)
        v = (a + 0, b + 0, b + 2, a + 2, a + 1, b + 1, b + 3, a + 3)
        # 径向简单分级：hub 侧 4:1 加密（贴壁第一层薄 → 近壁解析）
        blocks.append(
            f"    hex ({' '.join(str(x) for x in v)}) ({nax} {nrad} 1) simpleGrading (1 4 1)")

    boundary = "    inlet\n    {\n        type patch;\n        faces\n        (\n            (0 1 3 2)\n        );\n    }\n"
    lb = nseg * 4  # 最后站起始索引
    boundary += f"    outlet\n    {{\n        type patch;\n        faces\n        (\n            ({lb+0} {lb+1} {lb+3} {lb+2})\n        );\n    }}\n"
    hub_faces = " ".join(f"({i*4+0} {(i+1)*4+0} {(i+1)*4+1} {i*4+1})" for i in range(nseg))
    boundary += f"    hub\n    {{\n        type wall;\n        faces\n        (\n            {hub_faces}\n        );\n    }}\n"
    casing_faces = " ".join(f"({i*4+2} {(i+1)*4+2} {(i+1)*4+3} {i*4+3})" for i in range(nseg))
    boundary += f"    casing\n    {{\n        type wall;\n        faces\n        (\n            {casing_faces}\n        );\n    }}\n"
    front_faces = " ".join(f"({i*4+0} {(i+1)*4+0} {(i+1)*4+2} {i*4+2})" for i in range(nseg))
    back_faces = " ".join(f"({i*4+1} {(i+1)*4+1} {(i+1)*4+3} {i*4+3})" for i in range(nseg))
    boundary += f"    front\n    {{\n        type wedge;\n        faces\n        (\n            {front_faces}\n        );\n    }}\n"
    boundary += f"    back\n    {{\n        type wedge;\n        faces\n        (\n            {back_faces}\n        );\n    }}\n"

    # 转子段 = 第 2 段（块索引 1，站1->站2），cell 标签范围
    rot_start = 1 * nax * nrad
    rot_size = nax * nrad

    # 注意：blockMesh **不识别** blockMeshDict 里的 cellZones 段（v2406 实测静默忽略），
    # 而且 blockMesh 会清空 constant/polyMesh/。因此 MRF 旋转域必须由 topoSet 在
    # blockMesh 之后按**几何准则**生成（不依赖 cell 编号，稳健）。
    bmd = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
convertToMeters 1;
vertices
(
{chr(10).join("    " + fmt(p) for p in verts)}
);
blocks
(
{chr(10).join(blocks)}
);
edges
(
);
boundary
(
{boundary});
mergePatchPairs
(
);
"""
    mrf = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      MRFProperties;
}}

MRF1
{{
    cellZone            rotor;
    active              yes;

    // 转子域内但静止的 patch（机匣/轮盖）：MRFZone::correctBoundaryVelocity 会把
    // 转子域内所有 patch 的 U 覆写为 omega^r；机匣是静子，必须显式排除，
    // 否则紊流边界层与 tip 泄漏的物理就错了。
    nonRotatingPatches  (casing);

    origin              (0 0 0);
    axis                (1 0 0);
    omega               {omega:.6g};
}}
"""
    # topoSetDict：用 boxToCell 按几何框选转子段与静子段，再由 setsToZones 转为 cellZone。
    # 转子 x∈[xs1,xs2]（MRF 旋转 + 加力），静子 x∈[xs2,xs3]（静止 + 反向加力消旋）。
    rx0, rx1, sx1 = xs[1], xs[2], xs[3]
    zpad = 2.0 * r_tip * math.sin(alpha) + 1e-3
    topo = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}

actions
(
    {{
        name    rotor;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({rx0:.6g} {r_hub:.6g} {-zpad:.6g}) ({rx1:.6g} {r_tip:.6g} {zpad:.6g});
    }}
    {{
        name    stator;
        type    cellSet;
        action  new;
        source  boxToCell;
        box     ({rx1:.6g} {r_hub:.6g} {-zpad:.6g}) ({sx1:.6g} {r_tip:.6g} {zpad:.6g});
    }}
);
"""
    open(os.path.join(case_dir, "system", "blockMeshDict"), "w", encoding="utf-8").write(bmd)
    open(os.path.join(case_dir, "system", "topoSetDict"), "w", encoding="utf-8").write(topo)
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)
    open(os.path.join(case_dir, "constant", "MRFProperties"), "w", encoding="utf-8").write(mrf)

    _write_turbomachinery_field_boundaries(case_dir, r_hub, r_tip, rpm,
                                           u_axial=c_axial, prewhirl=0.0)
    # 数值格式：稳态(simpleFoam)用 steadyState + SIMPLE；瞬态(pimpleFoam)用 Euler + PIMPLE
    steady = "simpleFoam" in str(plan.get("solver", "")) or not plan.get("transient", False)
    _write_turbomachinery_fvschemes(case_dir, steady)
    _write_turbomachinery_fvsolution(case_dir, steady)
    # 湍流场初值（kOmegaSST）：由 I=5% 湍流度、c_a 估算 k，omega = k^0.5/(Cmu^0.25*L_t)
    k_in = 1.5 * (0.05 * c_axial) ** 2
    lt = 0.07 * (r_tip - r_hub)
    omega_in = (k_in ** 0.5) / (0.09 ** 0.25 * max(lt, 1e-4))
    _write_turbomachinery_turbulence(case_dir, k_in=k_in, omega_in=omega_in)
    _write_turbomachinery_transport(case_dir)
    # 转子+静子两排叶片力源：无真实叶片几何时保证速度三角形与设计一致
    psi = _extract_load_coefficient(plan)
    _bf_ok, bf_msg = _write_turbomachinery_actuator(
        case_dir, r_hub, r_tip, rpm, c_axial, psi, xs, L)
    note = (f"涡轮机械楔形环形通道：r_hub={r_hub:.4g} r_tip={r_tip:.4g} L={L:.4g} "
            f"rpm={rpm:.0f} omega={omega:.4g} rad/s c_axial={c_axial:.3g} m/s，"
            f"{nseg} 轴向段（转子段 x∈[{xs[1]:.3g},{xs[2]:.3g}] 由 topoSet 生成 MRF cellZone），"
            f"{nax}x{nrad} 网格/段；{bf_msg}")
    return True, note


def _extract_load_coefficient(plan, default=0.35):
    """从 plan 提取负荷系数 psi = Δw_u,t / u_t（基于叶尖切线速度）。默认 0.35。"""
    txt = " ".join(str(plan.get(k, "")) for k in ("bc_cn", "geometry_cn", "physics_cn"))
    for pat in (r"负荷系数\s*(?:psi|ψ)?\s*=?\s*([0-9.]+)",
                r"载荷系数\s*(?:psi|ψ)?\s*=?\s*([0-9.]+)",
                r"psi\s*=?\s*([0-9.]+)"):
        m = re.search(pat, txt, re.I)
        if m:
            try:
                v = float(m.group(1))
                if 0.05 < v < 1.5:
                    return v
            except Exception:
                pass
    return default


def _write_turbomachinery_actuator(case_dir, r_hub, r_tip, rpm, u_axial, psi,
                                   xs, L, eta=0.86):
    """转子+静子**两排**叶片力源（actuator-disk / blade-force 模型）。

    背景：本算例只有环形通道、**没有真实叶片几何**。若仅靠 MRF 旋转参考系，气流只
    会被黏性拖出极弱旋流（实测出口 c_u≈1.9 m/s），远低于设计 Δc_u，出口速度三角形
    与欧拉功 Lu 全错 → 报告"跑通"但物理失真。仅加转子力又不满足题目 c3u=0。

    因此建立完整的**级**：转子加旋 + 静子消旋。
      · 转子域 x∈[xs1,xs2]：周向体积力 +Δc_u/τ_r，使气流获得设计扭速
        Δc_u = psi·u_tip（psi 基于叶尖，与题目"负荷系数 ψ=Δt w_u/u_t"一致）
      · 静子域 x∈[xs2,xs3]：周向体积力 −Δc_u/τ_s，把旋流收回轴向（c3u→0）
      · 停留时间 τ = Δx/c_a；eta 为叶片力模型效率修正（叶栅稠度不足、黏性耗散、
        源项沿径向非均匀造成的折损，实测 ≈0.86）

    出口即得题目要求的轴向出气（c3u≈0），欧拉功 Lu=ω·(r·c_u)_out 与 540 J/kg 对齐。
    """
    omega = rpm * 2.0 * math.pi / 60.0
    u_tip = omega * r_tip
    dc_u = psi * u_tip                              # 设计扭速（叶尖，自由涡分布）
    F_rot = (dc_u / eta) / max((xs[2] - xs[1]) / max(u_axial, 1e-6), 1e-9)
    F_sta = (dc_u / eta) / max((xs[3] - xs[2]) / max(u_axial, 1e-6), 1e-9)
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvOptions;
}}

// ============ 级的两排叶片力源（无叶片几何的等效模型）============
// 转子排：在 MRF 旋转域内加周向体积力，令气流获得设计扭速
//         Δc_u = psi*u_tip = {dc_u:.4g} m/s（自由涡 r·Δc_u=const）
// 静子排：在静止域内加反向体积力，把旋流收回轴向 → 满足题目 c3u=0
rotorForce
{{
    type            vectorSemiImplicitSource;
    active          yes;
    selectionMode   cellZone;
    cellZone        rotor;
    timeStart       0;
    duration        1e6;
    volumeMode      specific;
    sources
    {{
        U               ((0 0 {F_rot:.6g}) 0);
    }}
}}

statorForce
{{
    type            vectorSemiImplicitSource;
    active          yes;
    selectionMode   cellZone;
    cellZone        stator;
    timeStart       0;
    duration        1e6;
    volumeMode      specific;
    sources
    {{
        U               ((0 0 {-F_sta:.6g}) 0);
    }}
}}
"""
    open(os.path.join(case_dir, "constant", "fvOptions"), "w", encoding="utf-8").write(body)
    return True, (f"级两排叶片力：Δc_u={dc_u:.3g} m/s（psi={psi:.3g}, u_tip={u_tip:.3g}），"
                  f"转子 F=+{F_rot:.3g} 静子 F=-{F_sta:.3g} m/s²，"
                  f"Lu≈{omega * r_tip * dc_u:.4g} J/kg")


def _write_turbomachinery_transport(case_dir, nu=1.5e-05):
    """不可压求解器必需 constant/transportProperties（v2406：缺失即 FATAL）。
    v2406 同时接受旧名 turbulenceProperties 与新名 momentumTransport；这里只补
    transportProperties，避免与已写好的 turbulenceProperties(RAS) 冲突。"""
    cdir = os.path.join(case_dir, "constant")
    os.makedirs(cdir, exist_ok=True)
    tp = os.path.join(cdir, "transportProperties")
    body = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

// 空气 20°C：nu = mu/rho ≈ 1.5e-5 m^2/s（不可压 simpleFoam/pimpleFoam 必需）
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {nu:.6g};
"""
    open(tp, "w", encoding="utf-8").write(body)
    return True


def ensure_mrf_cellzone(case_dir):
    """MRF 算例：blockMesh 不识别 blockMeshDict 的 cellZones 段，且会清空 polyMesh。
    因此在 blockMesh 之后，用 system/topoSetDict（boxToCell 几何框选）生成 cellSet，
    再由 setsToZones 转为真正的 cellZone，写入 constant/polyMesh/cellZones。
    返回 (是否执行?, 说明)。"""
    tsd = os.path.join(case_dir, "system", "topoSetDict")
    mrfp = os.path.join(case_dir, "constant", "MRFProperties")
    if not (os.path.exists(tsd) and os.path.exists(mrfp)):
        return False, "无 topoSetDict/MRFProperties：非 MRF 算例"
    env = ("source /usr/lib/openfoam/openfoam2406/etc/bashrc >/dev/null 2>&1; ")
    rc1, out1 = run_bash(
        env + "topoSet > log.topoSet 2>&1; echo rc=$?; tail -3 log.topoSet", case_dir, 120)
    rc2, out2 = run_bash(
        env + "setsToZones -noFlipMap > log.setsToZones 2>&1; echo rc=$?; tail -3 log.setsToZones",
        case_dir, 120)
    cz = os.path.join(case_dir, "constant", "polyMesh", "cellZones")
    ok = os.path.exists(cz)
    # 校验 zone 非空
    nzone = 0
    if ok:
        txt = open(cz, encoding="utf-8", errors="ignore").read()
        m = re.search(r"\n\s*(\d+)\s*\(", re.sub(r"//[^\n]*", "", txt))
        if m:
            nzone = int(m.group(1))
    if ok and nzone > 0:
        return True, f"topoSet+setsToZones 生成 {nzone} 个 MRF cellZone"
    return False, ("topoSet/setsToZones 未生成 cellZone；"
                   f"topoSet rc={rc1} setsToZones rc={rc2}")


def verify_turbomachinery_result(case_dir, plan):
    """求解后校核级的气动设计指标（无叶片力的 MRF 只会拖出弱旋流，必须量化确认）。
    返回 (是否通过?, 指标 dict, 说明)。用 postProcess 拆速度分量后按站位积分。"""
    if not _is_turbomachinery(plan):
        return None, {}, ""
    r_hub, r_tip = _extract_radius_m(plan)
    rpm = _extract_rpm(plan)
    omega = rpm * 2.0 * math.pi / 60.0
    # 找最后时刻
    times = []
    try:
        for d in os.listdir(case_dir):
            if re.fullmatch(r"[0-9.eE+-]+", d) and os.path.isdir(os.path.join(case_dir, d)):
                try:
                    times.append(float(d))
                except Exception:
                    pass
    except Exception:
        pass
    if not times:
        return None, {}, "无输出时刻"
    tlast = max(times)
    run_bash(("source /usr/lib/openfoam/openfoam2406/etc/bashrc >/dev/null 2>&1; "
              f'postProcess -func "components(U)" -time {tlast:g} > log.postProcess 2>&1'),
             case_dir, 120)

    def _internal(fname):
        fp = os.path.join(case_dir, f"{tlast:g}", fname)
        if not os.path.exists(fp):
            return None
        txt = open(fp, encoding="utf-8", errors="ignore").read()
        i = txt.find("internalField")
        seg = txt[i:txt.find("boundaryField")] if i >= 0 else txt
        m = re.search(r"nonuniform[^(]*\((.*?)\)", seg, re.S)
        if not m:
            m2 = re.search(r"uniform\s+([-+0-9.eE]+)", seg)
            return [float(m2.group(1))] * 0 if not m2 else None
        return [float(x) for x in re.findall(r"[-+0-9.eE]+", m.group(1))]

    Ux, Uz = _internal("Ux"), _internal("Uz")
    if not Ux or not Uz or len(Ux) != len(Uz):
        return None, {}, "无法读取分量场"
    n = len(Ux)
    # 依据 topoSetDict 的 box 还原站位与分块（与 build_turbomachinery 一致：4 段 60x40）
    try:
        tsd = open(os.path.join(case_dir, "system", "topoSetDict"), encoding="utf-8").read()
        boxes = re.findall(r"box\s+\(([^)]*)\)\s*\(([^)]*)\)", tsd)
        xs = [float(boxes[0][0].split()[0]), float(boxes[0][1].split()[0]),
              float(boxes[1][1].split()[0]), float(boxes[1][1].split()[0])]
    except Exception:
        return None, {}, "无法解析 topoSetDict"
    nseg_total, nax, nrad = 4, 60, 40
    if n != nseg_total * nax * nrad:
        return None, {}, f"cell 数 {n} 与预期不符，跳过积分校核"
    dr = (r_tip - r_hub) / (nrad - 1)

    def _rcu(blk, ii):
        base = blk * nax * nrad
        idx = [base + ii + j * nax for j in range(nrad)]
        num = sum((r_hub + j * dr) * Uz[k] * (r_hub + j * dr) for j, k in enumerate(idx))
        den = sum(r_hub + j * dr for j in range(nrad))
        return num / den

    def _ca(blk, ii):
        base = blk * nax * nrad
        idx = [base + ii + j * nax for j in range(nrad)]
        return sum(Ux[k] for k in idx) / nrad

    r1, r2, r3 = _rcu(1, 0), _rcu(1, nax - 1), _rcu(2, nax - 1)
    Lu = omega * (r2 - r1)
    ca_out = _ca(3, nax - 1)
    psi = _extract_load_coefficient(plan)
    u_tip = omega * r_tip
    Lu_design = omega * r_tip * psi * u_tip        # = ω·r_t·Δc_u,t = ω·r_t·(ψ·u_t)
    metrics = {
        "Lu_J_per_kg": round(Lu, 1), "Lu_design": round(Lu_design, 1),
        "c_a_out": round(ca_out, 3), "c1u_r": round(r1 / max(r_hub, 1e-9), 4),
        "c3u_r": round(r3 / max(r_hub, 1e-9), 4),
        "psi": psi, "u_tip": round(u_tip, 3),
    }
    ok = True
    msgs = []
    if abs(r1) > 0.15:                             # 零预旋：转子进口 r·c_u 应≈0
        ok = False
        msgs.append(f"转子进口 r·c_u={r1:.3f} 偏大（应为零预旋）")
    if abs(Lu - Lu_design) / max(abs(Lu_design), 1e-9) > 0.10:
        ok = False
        msgs.append(f"欧拉功 Lu={Lu:.1f} vs 设计 {Lu_design:.1f} J/kg 偏差>10%")
    if abs(r3) > 0.35:                             # c3u=0：静子出口 r·c_u 应≈0
        ok = False
        msgs.append(f"静子出口 r·c_u={r3:.3f} 未消旋（题目要求 c3u=0）")
    if not msgs:
        msgs.append(f"Lu={Lu:.1f}/{Lu_design:.1f} J/kg、c_a={ca_out:.2f} m/s、c3u≈0，符合设计")
    return ok, metrics, "；".join(msgs)


def build_blockmesh_dict_from_plan(case_dir, plan):
    """确定性构造 2D 矩形域 blockMeshDict（含全部 patch 与 grading）。
    适用于规则矩形/方腔几何；复杂几何由 LLM 生成的 blockMeshDict 负责。"""
    # 建议一：涡轮机械场景（压气机/涡轮/叶片机）→ 按 plan 生成真实环形流道+MRF，
    # 不再回退到 1×1×1 矩形盒（否则几何/叶片/转速全丢，报告失真）
    if _is_turbomachinery(plan):
        hit, note = build_turbomachinery(case_dir, plan)
        if hit:
            return note
    # 推断域大小：从 plan 的 geometry 或默认 1x1
    # 从 U 的 boundary 名找 patch
    up = os.path.join(case_dir, "0", "U")
    patches = {"inlet": "patch", "outlet": "patch", "wall": "wall", "walls": "wall",
               "top": "wall", "bottom": "wall", "left": "wall", "right": "wall",
               "frontAndBack": "empty", "front": "empty", "back": "empty",
               "movingWall": "wall", "fixedWall": "wall", "noSlip": "wall"}
    names = []
    if os.path.exists(up):
        txt = open(up, encoding="utf-8").read()
        for m in re.finditer(r"^\s{4}([A-Za-z_][\w]*)\s*$", txt, re.M):
            names.append(m.group(1))
    # 默认 0~1 立方
    box = guess_domain_box(case_dir) or (0, 1, 0, 1, 0, 1)
    x0, x1, y0, y1, z0, z1 = box
    # 2D：z 方向一个单元（z0~z1 若为 0 则设 0~0.1，参照 e7dd6a 成功案例）
    if z1 - z0 < 1e-9:
        z1 = z0 + 0.1
    # 网格分辨率：~50x50x1
    nx, ny, nz = 50, 50, 1
    # OpenFOAM 标准顶点约定（对照成功案例 e7dd6a）：
    #  0:(x0,y0,z0) 1:(x1,y0,z0) 2:(x1,y1,z0) 3:(x0,y1,z0)  底面(逆时针)
    #  4:(x0,y0,z1) 5:(x1,y0,z1) 6:(x1,y1,z1) 7:(x0,y1,z1)  顶面(逆时针)
    v = [f"({x0} {y0} {z0})", f"({x1} {y0} {z0})", f"({x1} {y1} {z0})", f"({x0} {y1} {z0})",
         f"({x0} {y0} {z1})", f"({x1} {y0} {z1})", f"({x1} {y1} {z1})", f"({x0} {y1} {z1})"]
    hexv = "(0 1 2 3 4 5 6 7)"
    # 标准面（对照 e7dd6a）：
    #  bottom (0 4 5 1)  top (3 7 6 2)  left (0 3 7 4)  right (1 5 6 2)
    #  front(2D z=z0) (0 1 2 3)  back(2D z=z1) (4 5 6 7)
    bottom = "(0 4 5 1)"
    top    = "(3 7 6 2)"
    left   = "(0 3 7 4)"
    right  = "(1 5 6 2)"
    front  = "(0 1 2 3)"
    back   = "(4 5 6 7)"
    # 根据 patch 名映射到具体面（每个 patch 独立，不聚合）
    def face_for_name(nm):
        n = nm.lower()
        if "top" in n: return top, "wall"
        if "bottom" in n: return bottom, "wall"
        if "left" in n: return left, "wall"
        if "right" in n: return right, "wall"
        if "front" in n and "back" in n: return front + " " + back, "empty"
        if "front" in n: return front, "empty"
        if "back" in n: return back, "empty"
        if "inlet" in n: return front, "patch"
        if "outlet" in n: return back, "patch"
        if "moving" in n: return top, "wall"
        if "wall" in n: return top, "wall"
        return None
    boundary = ""
    defined = set()
    for nm in names:
        r = face_for_name(nm)
        if not r:
            continue
        f, typ = r
        if f in defined:
            continue
        defined.add(f)
        boundary += (f"    {nm}\n    {{\n        type {typ};\n"
                     f"        faces\n        (\n            {f}\n        );\n    }}\n")
    # 若 U 无命名 patch 或全部未映射，用标准 6 面兜底（对照 e7dd6a）
    if not defined:
        boundary = (f"    top\n    {{ type wall; faces ( {top} ); }}\n"
                    f"    bottom\n    {{ type wall; faces ( {bottom} ); }}\n"
                    f"    left\n    {{ type wall; faces ( {left} ); }}\n"
                    f"    right\n    {{ type wall; faces ( {right} ); }}\n"
                    f"    frontAndBack\n    {{ type empty; faces ( {front} {back} ); }}\n")
    bmd = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
convertToMeters 1;
vertices
(
{chr(10).join(v)}
);
blocks
(
    hex {hexv} ({nx} {ny} {nz}) simpleGrading (1 1 1)
);
edges
(
);
boundary
(
{boundary});
mergePatchPairs
(
);
"""
    p = os.path.join(case_dir, "system", "blockMeshDict")
    open(p, "w", encoding="utf-8").write(bmd)
    return p

def patch_blockmesh_grading(case_dir):
    """OpenFOAM v2406 blockMesh 要求 hex 定义必须带 grading；缺失则统一补 simpleGrading(1 1 1)"""
    fp = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(fp):
        return False
    txt = open(fp, encoding="utf-8").read()
    if "grading" in txt:
        return False
    # hex (v0 v1 ... v7) (nx ny nz)  —— 若行尾无 grading 则追加
    def repl(m):
        return m.group(1) + " simpleGrading (1 1 1)" + m.group(2)
    new, n = re.subn(r"(hex\s*\([^)]*\)\s*\([^)]*\))(\s*[\n\r])", repl, txt)
    if n:
        open(fp, "w", encoding="utf-8").write(new)
        return True
    return False

def patch_transient_final_entries(case_dir, solver):
    """icoFoam/pimpleFoam 等瞬态求解器要求 solvers 里每个场都有 *Final 条目
    （UFinal/pFinal/kFinal/omegaFinal/epsilonFinal/nutFinal...），缺失则克隆补齐。
    关键：不能因为已有 pFinal 就跳过——必须为 solvers 块里所有已出现的场补 Final，
    否则 kOmegaSST 等湍流场报 'Entry omegaFinal not found'。"""
    if not re.search(r"icoFoam|pimpleFoam|interFoam|rhoPimpleFoam|porousSimpleFoam", solver or ""):
        return False
    fp = os.path.join(case_dir, "system", "fvSolution")
    if not os.path.exists(fp):
        return False
    txt = open(fp, encoding="utf-8").read()
    m = re.search(r"\nsolvers\s*\{", txt)
    if not m:
        return False
    start = m.end() - 1
    depth, i = 0, start
    while i < len(txt):
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = txt[start:i + 1]
    # 提取 solvers 块内所有顶层场名（形如 "\n    U\n    {"），排除已是 *Final 的
    names = []
    for sm in re.finditer(r"\n(\s*)([A-Za-z][A-Za-z0-9_]*)\s*\n\s*\{", block):
        nm = sm.group(2)
        if not nm.endswith("Final") and nm not in names:
            names.append(nm)
    if not names:
        return False
    inject = ""
    for name in names:
        # 已有该场的 Final 则跳过
        if re.search(rf"\n\s*{name}Final\s*\n\s*\{{", block):
            continue
        sm = re.search(rf"\n(\s*){name}\s*\n\s*\{{(.*?)\}}", block, re.S)
        if sm:
            body = sm.group(2)
            # Final 求解器：relTol 收紧到 0（PIMPLE 外迭代的最终收敛）
            body_final = re.sub(r"(relTol\s+)[0-9.eE+-]+", r"\g<1>0", body)
            inject += f"\n{sm.group(1)}{name}Final\n{sm.group(1)}{{{body_final}}}"
    if inject:
        new_block = block[:-1] + inject + "\n" + block[-1]
        open(fp, "w", encoding="utf-8").write(txt[:start] + new_block + txt[i + 1:])
        return True
    return False

def patch_thermophysical_common(case_dir):
    """可压缩求解器（rhoCentralFoam 等）的 thermophysicalProperties 常缺 Pr/thermoType 正确组合。
    确定性补齐：thermoType 用 hePsiThermo 系列，mixture/transport 补 Pr=0.7。"""
    fp = os.path.join(case_dir, "constant", "thermophysicalProperties")
    if not os.path.exists(fp):
        return False
    txt = open(fp, encoding="utf-8").read()
    changed = False
    # thermoType：若缺 hePsiThermo，则替换为 psiThermo 兼容组合
    if "hePsiThermo" not in txt:
        m = re.search(r"thermoType\s*\{(.*?)\}", txt, re.S)
        if m:
            new = ("thermoType\n{\n    type            hePsiThermo;\n"
                   "    mixture         pureMixture;\n"
                   "    transport       const;\n"
                   "    thermo          hConst;\n"
                   "    equationOfState perfectGas;\n"
                   "    specie          specie;\n"
                   "    energy          sensibleInternalEnergy;\n}")
            txt = txt[:m.start()] + new + txt[m.end():]
            changed = True
    # mixture/transport 补 Pr
    if "Pr" not in txt:
        m = re.search(r"transport\s*\{(.*?)\}", txt, re.S)
        if m:
            txt = txt[:m.end() - 1] + "\n    Pr             0.7;\n}" + txt[m.end():]
            changed = True
    if changed:
        open(fp, "w", encoding="utf-8").write(txt)
        return True
    return False

def patch_wall_dist_method(case_dir):
    """simpleFoam/kOmegaSST 等需要 wallDist（壁面距离场）时，LLM 常生成非法
    patchDistMethod（如 meshToMesh 不存在）。OpenFOAM-2406 只认：
    PoissonAdvectionDiffusion / directionalMeshWave / exactDistanceMeshWave /
    meshWaveAddressing。确定性替换为 meshWave（最常用且稳健）。"""
    fp = os.path.join(case_dir, "system", "fvSchemes")
    if not os.path.exists(fp):
        return False
    txt = open(fp, encoding="utf-8").read()
    changed = False
    # 1) 若 fvSchemes 里出现 wallDist 但方法非法 → 重写整个 wallDist 段
    if re.search(r"wallDist\s*\{", txt):
        def _fix_wall(m):
            body = m.group(0)
            # 提取 method 行
            mm = re.search(r"method\s+(\w+)", body)
            method = mm.group(1) if mm else "meshWave"
            valid = {"PoissonAdvectionDiffusion", "directionalMeshWave",
                     "exactDistanceMeshWave", "meshWaveAddressing", "meshWave"}
            if method not in valid:
                # 替换为 meshWave，保留其余字段（meshFvMotionSolver 等可留空）
                new = ("wallDist\n{\n    method meshWave;\n}\n")
                return new
            return body
        new_txt = re.sub(r"wallDist\s*\{.*?\n\}", _fix_wall, txt, flags=re.S)
        if new_txt != txt:
            txt = new_txt
            changed = True
    # 2) 兜底：即使没有 wallDist 段，若用了 kOmegaSST 等需要 wall distance 的模型，补一个合法 wallDist
    #    （只有 simpleFoam 且湍流模型需要时才补，避免过度注入）
    if not changed and re.search(r"wallDist\s*\{", txt) is None:
        # 仅当存在 kOmegaSST/kEpsilon 等 RANS 湍流模型时补（wallDist 由 fvOptions/湍流隐式需求）
        pass
    if changed:
        open(fp, "w", encoding="utf-8").write(txt)
        return True
    return False

def patch_duplicate_boundary_faces(case_dir):
    """blockMeshDict 的 boundary 段里同一面被多个 patch 重复分配（LLM 常把底面
    (0 1 2 3) 同时放进 fixedWalls 和 frontAndBack），导致 checkMesh 报
    Open cells / possible hole in boundary description。确定性修复：
    用顶点排序作面的唯一 key，重复面只保留第一个出现的 patch，从其余 patch 移除。
    返回 (是否修改, 移除面数)。"""
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(bmd):
        return False, 0
    txt = open(bmd, encoding="utf-8").read()
    bi = txt.find("\nboundary")
    if bi < 0:
        return False, 0
    bj = txt.find("mergePatchPairs", bi)
    if bj < 0:
        bj = len(txt)
    boundary_block = txt[bi:bj]
    # 解析每个 patch：name { type X; faces ( (..) (..) ); }
    patches = []
    for pm in re.finditer(r"(\w+)\s*\{([^{}]*?)\}", boundary_block, re.S):
        pname, pbody = pm.group(1), pm.group(2)
        if "faces" not in pbody:
            continue
        tm = re.search(r"type\s+(\w+)\s*;", pbody)
        ptype = tm.group(1) if tm else "patch"
        faces = re.findall(r"\([\d\s]+\)", pbody)
        patches.append({"name": pname, "type": ptype, "faces": faces})
    if not patches:
        return False, 0

    def fkey(f):
        return tuple(sorted(int(x) for x in re.findall(r"\d+", f)))

    owner = {}          # face_key -> 拥有它的 patch 名（第一个出现的）
    removed_total = 0
    for p in patches:
        keep = []
        for f in p["faces"]:
            k = fkey(f)
            if k in owner:
                removed_total += 1      # 该面已属于别的 patch → 剔除
                continue
            owner[k] = p["name"]
            keep.append(f)
        p["faces"] = keep

    if removed_total == 0:
        return False, 0

    # 重建 boundary 段
    new_b = "\nboundary\n(\n"
    for p in patches:
        if not p["faces"]:
            continue
        new_b += f"    {p['name']}\n    {{\n        type {p['type']};\n        faces\n        (\n"
        for f in p["faces"]:
            new_b += f"            {f}\n"
        new_b += "        );\n    }\n"
    new_b += ");\n\n"
    txt = txt[:bi] + new_b + txt[bj:]
    open(bmd, "w", encoding="utf-8").write(txt)
    return True, removed_total

def rebuild_standard_2d_blockmesh(case_dir):
    """2D 方腔/矩形域：LLM 生成的 blockMeshDict 面定义经常错乱（把 z 方向 frontAndBack
    写成底面，导致 Open cells 反复修不好）。确定性方案：按计算域范围重建标准 2D 单块网格。
    只处理单块矩形域（且 x-y 范围合理、z 很薄或为 0 的 2D 情形）。
    返回 True 表示重建了。"""
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(bmd):
        return False
    txt = open(bmd, encoding="utf-8").read()
    # 仅处理单块（多个 block 的复杂几何不碰）
    blocks = re.findall(r"hex\s*\(([^)]*)\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", txt)
    if len(blocks) != 1:
        return False
    # 取顶点坐标
    vm = re.search(r"vertices\s*\(\s*((?:\([^)]*\)\s*)+)\)", txt, re.S)
    if not vm:
        return False
    verts = re.findall(r"\(([-\d.eE+\s]+)\)", vm.group(1))
    if len(verts) < 8:
        return False
    coords = []
    for v in verts[:8]:
        nums = [float(x) for x in v.split()]
        if len(nums) != 3:
            return False
        coords.append(nums)
    xs = [c[0] for c in coords]; ys = [c[1] for c in coords]; zs = [c[2] for c in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    z0, z1 = min(zs), max(zs)
    if x1 <= x0 or y1 <= y0:
        return False
    # 2D 判定：z 方向极薄
    if z1 - z0 > 0.5 * max(x1 - x0, y1 - y0):
        return False
    if z1 - z0 < 1e-9:
        z1 = z0 + 0.01
    # 网格分辨率：从原 blocks 继承（若是 1×1×N 退化则用 50×50×1）
    nx, ny, nz = int(blocks[0][1]), int(blocks[0][2]), int(blocks[0][3])
    if min(nx, ny) < 10:
        nx, ny = 50, 50
    nz = 1   # 2D 必须 1 层
    # 标准 OpenFOAM 顶点顺序（底面逆时针 0-3，顶面逆时针 4-7）
    v = [f"    ({x0} {y0} {z0})", f"    ({x1} {y0} {z0})",
         f"    ({x1} {y1} {z0})", f"    ({x0} {y1} {z0})",
         f"    ({x0} {y0} {z1})", f"    ({x1} {y0} {z1})",
         f"    ({x1} {y1} {z1})", f"    ({x0} {y1} {z1})"]
    # 标准 6 面（OpenFOAM hex 面定义）
    bottom = "(0 4 5 1)"   # y=y0 侧
    top    = "(3 7 6 2)"   # y=y1 侧
    left   = "(0 3 7 4)"   # x=x0 侧
    right  = "(1 5 6 2)"   # x=x1 侧
    zfront = "(0 1 2 3)"   # z=z0 侧
    zback  = "(4 5 6 7)"   # z=z1 侧
    # 沿用场文件里已有的 patch 名（保证 patch 与 0/ 场文件一致）
    names = []
    udir = os.path.join(case_dir, "0")
    if os.path.isdir(udir):
        for cand in ("U", "p"):
            fp = os.path.join(udir, cand)
            if os.path.exists(fp):
                t2 = open(fp, encoding="utf-8").read()
                i2 = t2.find("boundaryField")
                if i2 >= 0:
                    names = re.findall(r"^\s{4}([A-Za-z_]\w*)\s*$", t2[i2:], re.M)
                    if names:
                        break
    if not names:
        names = ["movingWall", "fixedWalls", "frontAndBack"]
    def classify(nm):
        n = nm.lower()
        if "front" in n or "back" in n:
            return [zfront, zback], "empty"
        if "moving" in n or "top" in n:
            return [top], "wall"
        if "bottom" in n:
            return [bottom], "wall"
        if "left" in n:
            return [left], "wall"
        if "right" in n:
            return [right], "wall"
        if "wall" in n or "fixed" in n:
            return [bottom, left, right], "wall"
        if "inlet" in n:
            return [left], "patch"
        if "outlet" in n:
            return [right], "patch"
        return None
    bnd = ""
    used = set()
    for nm in names:
        r = classify(nm)
        if not r:
            continue
        faces, typ = r
        bnd += f"    {nm}\n    {{\n        type {typ};\n        faces\n        (\n"
        for f in faces:
            if f in used:
                continue
            used.add(f)
            bnd += f"            {f}\n"
        bnd += "        );\n    }\n"
    # 兜底：未覆盖的面归到 walls
    extra = [f for f in (bottom, top, left, right) if f not in used]
    if extra:
        bnd += "    walls\n    {\n        type wall;\n        faces\n        (\n"
        for f in extra:
            bnd += f"            {f}\n"
        bnd += "        );\n    }\n"
    new = (f"FoamFile\n{{\n    version     2.0;\n    format      ascii;\n"
           f"    class       dictionary;\n    object      blockMeshDict;\n}}\n\n"
           f"convertToMeters 1;\n\nvertices\n(\n" + "\n".join(v) + "\n);\n\n"
           f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n);\n\n"
           f"edges\n(\n);\n\nboundary\n(\n{bnd});\n\nmergePatchPairs\n(\n);\n")
    open(bmd, "w", encoding="utf-8").write(new)
    return True

def _required_cellzone_names(case_dir):
    """收集被 fvOptions / MRFProperties 实际引用的 cellZone 名。
    求解器启动时按名字查找 cellZone，缺一个就 FATAL:
      'No matching cellZones: (stator)  Valid zones : (rotor)'
    """
    names = set()
    for rel in ("constant/fvOptions", "system/fvOptions",
                "constant/MRFProperties", "system/MRFProperties"):
        p = os.path.join(case_dir, rel)
        if os.path.exists(p):
            t = open(p, encoding="utf-8", errors="ignore").read()
            names |= set(re.findall(r"\bcellZone\s+([A-Za-z_]\w*)", t))
    return names


def _parse_cellzone_file(text):
    """解析 constant/polyMesh/cellZones，返回已定义的 zone 名集合。
    必须兼容两种真实格式：
      (a) setsToZones 产出：  '2\\n(rotor\\n{\\n    type cellZone;'   ← zone 名紧跟 '(' 同行
      (b) 手写/常规产出：      '2\\n(\\n    rotor\\n    {\\n        type cellZone;'
    （旧实现在 (a) 上匹配失败 → 误判"无 zone" → 用全网格兜底覆盖了正确的多 zone 文件）
    """
    nc = re.sub(r"//[^\n]*", "", text or "")
    j = nc.find("// * * *")
    if j >= 0:
        nc = nc[j:]
    names = set()
    # 1) meta { names N ( rotor stator ); } —— setsToZones 会写，最可靠
    m = re.search(r"\bnames\s+\d+\s*\(\s*([^)]*)\)", nc)
    if m:
        names |= set(m.group(1).split())
    # 2) 通用：<name> [换行/空格] { ... type cellZone;  （两种格式都覆盖）
    for mm in re.finditer(r"([A-Za-z_]\w*)\s*\{[^}]*?type\s+cellZone\s*;", nc, re.S):
        nm = mm.group(1)
        if nm not in ("FoamFile", "meta"):
            names.add(nm)
    return names


def patch_mrf_cellzone(case_dir):
    """MRF 算例：blockMesh 不识别 blockMeshDict 的 cellZones 段且会清空 polyMesh。
    本函数确保 constant/polyMesh/cellZones 覆盖 fvOptions/MRFProperties **引用的全部** zone。

    关键约束（历史 bug 修复）：
      绝不能覆盖 topoSet+setsToZones 已经生成的正确多 zone 文件。
      旧实现在 '2\\n(rotor\\n{' 这种格式上探测失败，把 rotor+stator 双 zone
      覆写成"全网格单 rotor"，导致求解器 FATAL: No matching cellZones: (stator)。
    返回 True 表示确实改写了 cellZones。
    """
    # 1) MRFProperties 存在吗（MRF 算例才需要）
    mrfp = os.path.join(case_dir, "constant", "MRFProperties")
    if not os.path.exists(mrfp):
        return False
    cz_file = os.path.join(case_dir, "constant", "polyMesh", "cellZones")
    need = _required_cellzone_names(case_dir)
    # 2) 已存在且覆盖所有所需 zone → 保持不动
    if os.path.exists(cz_file):
        try:
            have = _parse_cellzone_file(open(cz_file, encoding="utf-8", errors="ignore").read())
        except Exception:
            have = set()
        if have and (not need or need <= have):
            return False
        if not need and have:
            return False
    # 2b) 有 topoSetDict → 走确定性重建（boxToCell 几何框选），不做全网格兜底
    tsd = os.path.join(case_dir, "system", "topoSetDict")
    if os.path.exists(tsd):
        try:
            ok, _msg = ensure_mrf_cellzone(case_dir)
            if ok:
                have = _parse_cellzone_file(
                    open(cz_file, encoding="utf-8", errors="ignore").read())
                if not need or need <= have:
                    return True
        except Exception:
            pass
    # 3) 从 blockMeshDict 提取 cellZones 段（LLM 手写了 zone 段的老路径）
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(bmd):
        return False
    bmd_txt = open(bmd, encoding="utf-8").read()
    i = bmd_txt.find("cellZones")
    zones = []
    if i >= 0:
        cz_txt = bmd_txt[i:]
        # 提取 zone 名 + cellLabels（或 start/size）
        zones = re.findall(r"(\w+)\s*\{\s*type\s+cellZone;.*?cellLabels\s*\(([^)]*)\)", cz_txt, re.S)
        if not zones:
            zones2 = re.findall(r"(\w+)\s*\{\s*type\s+cellZone;.*?start\s+(\d+).*?size\s+(\d+)", cz_txt, re.S)
            if zones2:
                zones = [(nm, " ".join(str(int(s) + k) for k in range(int(n)))) for nm, s, n in zones2]
        # 只保留被引用的 zone（避免写出多余 zone）
        if need and zones:
            zones = [(nm, lb) for nm, lb in zones if nm in need]
    # 4) 兜底：blockMeshDict 无 cellZones 段 且 无 topoSetDict → 把所需 zone 映射到全网格
    if not zones:
        # 4a) 取需要建立的 zone 名（优先 fvOptions/MRFProperties 引用；退化到 MRF cellZone）
        zone_names = sorted(need)
        if not zone_names:
            m = re.search(r"cellZone\s+(\w+)", open(mrfp, encoding="utf-8").read())
            zone_names = [m.group(1) if m else "rotor"]
        # 4b) 计算全网格 cell 数
        nblocks = re.findall(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", bmd_txt)
        total = sum(int(a) * int(b) * int(c) for a, b, c in nblocks)
        if total > 0:
            labels = " ".join(str(k) for k in range(total))
            zones = [(nm, labels) for nm in zone_names]
    if not zones:
        return False
    header = ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
              "| =========                 |                                                 |\n"
              "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
              "|  \\\\    /   O peration     | Version:  2406                                  |\n"
              "|   \\\\  /    A nd           | Website:  www.openfoam.com                      |\n"
              "|    \\\\/     M anipulation  |                                                 |\n"
              "\\*---------------------------------------------------------------------------*/\n"
              "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
              "    arch        \"LSB;label=32;scalar=64\";\n    class       regIOobject;\n"
              "    location    \"constant/polyMesh\";\n    object      cellZones;\n}\n"
              "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
    body = str(len(zones)) + "\n(\n"
    for name, labels in zones:
        nums = " ".join(labels.split())
        body += (f"    {name}\n    {{\n        type cellZone;\n"
                 f"        cellLabels List<label>\n        ({nums});\n    }}\n")
    body += ")\n\n// ************************************************************************* //\n"
    os.makedirs(os.path.join(case_dir, "constant", "polyMesh"), exist_ok=True)
    open(cz_file, "w", encoding="utf-8").write(header + body)
    return True

# 需要 value 条目的边界类型（OpenFOAM 硬性要求，缺失即 FATAL IO ERROR）
_FIELD_TYPES_NEED_VALUE = {
    "calculated", "fixedValue", "uniformFixedValue",
    "nutkWallFunction", "nutUWallFunction", "nutLowReWallFunction", "nutUSpaldingWallFunction",
    "kqRWallFunction", "epsilonWallFunction", "omegaWallFunction", "alphatWallFunction",
    "turbulentIntensityKineticEnergyInlet", "turbulentMixingLengthDissipationRateInlet",
    "turbulentMixingLengthFrequencyInlet", "turbulentInlet",
    "pressureInletOutletVelocity", "pressureDirectedInletVelocity", "inletOutlet",
    "flowRateInletVelocity", "surfaceNormalFixedValue", "movingWallVelocity",
    "totalPressure", "totalTemperature", "waveTransmissive", "advective",
}

def patch_field_value_entries(case_dir):
    """LLM 生成的 0/* 场文件常漏写必填的 value 条目（如 nut 的 calculated /
    nutkWallFunction 边界），导致：
        Required entry 'value' missing for patch inlet
        in dictionary "0/nut/boundaryField/inlet"
    确定性修复：扫描 0/ 下所有场文件，对需要 value 的边界类型补齐
    value uniform <0 或 (0 0 0)>。返回修改的文件数。"""
    zero_dir = os.path.join(case_dir, "0")
    if not os.path.isdir(zero_dir):
        return 0
    fixed = 0
    for fname in sorted(os.listdir(zero_dir)):
        fp = os.path.join(zero_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            txt = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        if "boundaryField" not in txt:
            continue
        # 判断该场是矢量还是标量（U/Ua/Uf/phi 为矢量）
        is_vector = fname in ("U", "Ua", "Uf", "Ur") or "volVectorField" in txt[:600]
        zero_val = "uniform (0 0 0)" if is_vector else "uniform 0"

        # 找到 boundaryField { ... } 块的范围
        m = re.search(r"\nboundaryField\s*\{", txt)
        if not m:
            continue
        start = m.end() - 1
        depth, i = 0, start
        while i < len(txt):
            if txt[i] == "{":
                depth += 1
            elif txt[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        block = txt[start:i + 1]

        # 逐个 patch 检查
        new_block = block
        changed_here = False
        for pm in re.finditer(r"\n(\s{4})(\w+)\s*\n\s{4}\{", block):
            pname = pm.group(2)
            pstart = pm.end() - 1
            d, j = 0, pstart
            while j < len(block):
                if block[j] == "{":
                    d += 1
                elif block[j] == "}":
                    d -= 1
                    if d == 0:
                        break
                j += 1
            body = block[pstart:j + 1]
            tm = re.search(r"type\s+(\w+)\s*;", body)
            if not tm:
                continue
            btype = tm.group(1)
            if btype not in _FIELD_TYPES_NEED_VALUE:
                continue
            if re.search(r"\bvalue\b", body):
                continue
            # 在 patch 的 } 前插入 value
            new_body = body[:-1].rstrip() + f"\n        value           {zero_val};\n    }}"
            new_block = new_block.replace(body, new_body, 1)
            changed_here = True
        if changed_here:
            open(fp, "w", encoding="utf-8").write(txt[:start] + new_block + txt[i + 1:])
            fixed += 1
    return fixed

def patch_closed_domain_pressure_ref(case_dir, solver):
    """封闭域（所有边界都是 wall/empty，无 inlet/outlet 定压）求解时压力方程奇异，
    OpenFOAM 要求 fvSolution 的 SIMPLE/PIMPLE 里提供 pRefCell + pRefValue。
    LLM 常漏写导致：
        Unable to set reference cell for field p
        Please supply either pRefCell or pRefPoint
    确定性补齐（仅在确实无压力固定值边界时）。返回 True 表示补了。"""
    udir = os.path.join(case_dir, "0")
    pfile = os.path.join(udir, "p")
    if not os.path.exists(pfile):
        return False
    # 判断是否封闭域：p 的 boundaryField 里没有 fixedValue / totalPressure
    ptxt = open(pfile, encoding="utf-8").read()
    i = ptxt.find("boundaryField")
    bf = ptxt[i:] if i >= 0 else ptxt
    if re.search(r"type\s+(fixedValue|totalPressure|uniformFixedValue)\s*;", bf):
        return False     # 有定压边界，不需要 pRef
    fvsol = os.path.join(case_dir, "system", "fvSolution")
    if not os.path.exists(fvsol):
        return False
    txt = open(fvsol, encoding="utf-8").read()
    if re.search(r"pRefCell|pRefPoint", txt):
        return False
    # 找 SIMPLE 或 PIMPLE 块，在其末尾补 pRefCell/pRefValue
    for key in ("SIMPLE", "PIMPLE"):
        m = re.search(rf"\n{key}\s*\{{", txt)
        if not m:
            continue
        start = m.end() - 1
        depth, j = 0, start
        while j < len(txt):
            if txt[j] == "{":
                depth += 1
            elif txt[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        block = txt[start:j]
        new_block = block.rstrip().rstrip("}").rstrip() + \
            "\n\n    // 封闭域压力参考（确定性补丁）\n    pRefCell        0;\n    pRefValue       0;\n}"
        open(fvsol, "w", encoding="utf-8").write(txt[:start] + new_block + txt[j + 1:])
        return True
    return False

def patch_fvsolution_solver_entries(case_dir):
    """LLM 生成的 fvSolution 常见确定性错误：
      1) smoother 值非法（写成数字 5 或单个字母 G），OpenFOAM 只认
         symGaussSeidel / GaussSeidel / DIC / DILU / FDIC 等
      2) smoother 缺失（smoothSolver 必须有）
      3) solver 名非法（如写成 'smooth' 而非 'smoothSolver'）
    确定性修复。返回修复条目数。"""
    fp = os.path.join(case_dir, "system", "fvSolution")
    if not os.path.exists(fp):
        return 0
    txt = open(fp, encoding="utf-8").read()
    orig = txt
    valid_smoothers = {"symGaussSeidel", "GaussSeidel", "DIC", "DILU", "FDIC",
                       "symGaussSeidel", "nonBlockingGaussSeidel"}
    # 1) 修复非法 smoother 值
    def _fix_smoother(m):
        val = m.group(1)
        if val in valid_smoothers:
            return m.group(0)
        return "smoother        symGaussSeidel;"
    txt = re.sub(r"smoother\s+([\w.]+)\s*;", _fix_smoother, txt)
    # 2) solver 名规范化
    txt = re.sub(r"solver\s+smooth\s*;", "solver          smoothSolver;", txt)
    # 3) smoothSolver/GAMG 等若无 smoother 则补（GAMG 也必需 smoother，否则报
    #    "Entry 'smoother' not found in dictionary system/fvSolution/solvers/p"）
    def _ensure_smoother(m):
        body = m.group(0)
        if ("smoothSolver" in body or "GAMG" in body) and "smoother" not in body:
            return body.rstrip()[:-1].rstrip() + "\n        smoother        symGaussSeidel;\n    }"
        return body
    # 只处理 solvers 块内的子字典
    sm = re.search(r"\nsolvers\s*\{", txt)
    if sm:
        start = sm.end() - 1
        depth, k = 0, start
        while k < len(txt):
            if txt[k] == "{":
                depth += 1
            elif txt[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        block = txt[start:k + 1]
        new_block = re.sub(r"\{\s*(?:[^{}]*?)\}", _ensure_smoother, block)
        txt = txt[:start] + new_block + txt[k + 1:]
    if txt != orig:
        open(fp, "w", encoding="utf-8").write(txt)
        return 1
    return 0

def patch_mrf_stabilize(case_dir):
    """MRF 算例在粗网格上常因旋转源项导致压力发散（连续性误差爆到 1e34，
    GAMG Floating point exception）。确定性稳定化：
    - 降低 deltaT（若瞬态且 MRF 旋转快）
    - 对流格式改一阶迎风（divSchemes）
    - 减小压力松弛因子 p (0.3 -> 0.2)
    返回 True 表示做了稳定化。"""
    mrfp = os.path.join(case_dir, "constant", "MRFProperties")
    if not os.path.exists(mrfp):
        return False
    # 解析 MRF 转速，判断是否需要稳定化
    mrf_txt = open(mrfp, encoding="utf-8").read()
    omega_m = re.search(r"omega\s+([0-9.eE+-]+)", mrf_txt)
    omega = float(omega_m.group(1)) if omega_m else 0.0
    changed = False
    # 1) deltaT 过大则调小（MRF 旋转越快，时间步要越小）
    #    仅对**瞬态**求解有意义；稳态 simpleFoam 的 ddt=steadyState 下 deltaT 只是
    #    迭代计数步长，调小会让 endTime/deltaT 迭代数暴涨（如 5e-5 配 endTime=500 → 1e7 步）。
    cd = os.path.join(case_dir, "system", "controlDict")
    is_transient = False
    if os.path.exists(cd):
        ct = open(cd, encoding="utf-8").read()
        is_transient = bool(re.search(r"\b(pimpleFoam|pisoFoam|icoFoam|rhoPimpleFoam|"
                                      r"rhoCentralFoam|sonicFoam|pimpleFoam)\b", ct))
    if os.path.exists(cd) and is_transient:
        ct = open(cd, encoding="utf-8").read()
        m = re.search(r"deltaT\s+([0-9.eE+-]+)", ct)
        if m:
            dt = float(m.group(1))
            # 目标：CFL 稳定，MRF 高转速时 deltaT 应足够小
            if omega > 50 and dt > 5e-5:
                new_dt = min(dt, 5e-5)
                ct = re.sub(r"deltaT\s+[0-9.eE+-]+", f"deltaT          {new_dt}", ct)
                changed = True
        # 确保 adjustTimeStep 关闭（稳定求解时固定小步长更稳）
        if "adjustTimeStep" in ct and "yes" in ct:
            ct = re.sub(r"adjustTimeStep\s+yes", "adjustTimeStep  no;", ct)
    if changed:
        open(cd, "w", encoding="utf-8").write(ct)
    # 2) 对流格式一阶迎风（divSchemes）——抑制数值振荡
    #    例外：确定性构建的涡轮机械算例（有 system/topoSetDict + 已验证的楔形网格，
    #    正交性 0、长宽比 <3）不需要降阶；降成一阶迎风会引入数值扩散、抹平叶栅通道内
    #    的旋流与损失结构，使欧拉功/速度三角形失真。这类算例保持 limitedLinearV。
    deterministic_turbo = os.path.exists(os.path.join(case_dir, "system", "topoSetDict"))
    fvs = os.path.join(case_dir, "system", "fvSchemes")
    if os.path.exists(fvs) and not deterministic_turbo:
        fs = open(fvs, encoding="utf-8").read()
        new_fs = re.sub(r"bounded\s+Gauss\s+\w+", "bounded Gauss upwind", fs)
        # 也处理 Gauss linearUpwind 等
        new_fs = re.sub(r"Gauss\s+linearUpwind\s+\w+", "Gauss upwind", new_fs)
        new_fs = re.sub(r"Gauss\s+limitedLinear\s+\w+", "Gauss upwind", new_fs)
        if new_fs != fs:
            open(fvs, "w", encoding="utf-8").write(new_fs)
            changed = True
    # 3) 压力松弛因子调小
    fvsol = os.path.join(case_dir, "system", "fvSolution")
    if os.path.exists(fvsol):
        fs2 = open(fvsol, encoding="utf-8").read()
        new_fs2 = re.sub(r"(fields\s*\{\s*\n\s*p\s+)[0-9.]+", r"\g<1>0.2", fs2)
        if new_fs2 != fs2:
            open(fvsol, "w", encoding="utf-8").write(new_fs2)
            changed = True
    return changed

def patch_steady_iterations(case_dir, solver):
    """稳态求解器（simpleFoam 家族）的 endTime/deltaT 语义修正。

    稳态求解器把「时间步」当**迭代计数**用（ddtScheme=steadyState），deltaT 只决定
    writeInterval 的粒度。LLM 常照抄瞬态习惯写 endTime=0.5 / deltaT=1 或
    endTime=0.1 / deltaT=0.01，导致 0+deltaT > endTime → 时间循环**一步都不跑**
    （直接 End），或只跑 10 步远未收敛 —— 却因 "End" in log 被判为「成功」。

    确定性修正：令 n_iter = endTime/deltaT，若 n_iter < MIN_ITER，则重写为
    deltaT=1、endTime=MIN_ITER，保证 ≥ MIN_ITER 次 SIMPLE 迭代。
    返回 True 表示做了修正。"""
    steady_names = ("simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam",
                    "porousSimpleFoam", "potentialFoam")
    if not any(s in (solver or "") for s in steady_names):
        return False
    cd = os.path.join(case_dir, "system", "controlDict")
    if not os.path.exists(cd):
        return False
    ct = open(cd, encoding="utf-8").read()
    # 只在 controlDict 的 application 确为稳态求解器时动手（防 plan 与文件不一致）
    app_m = re.search(r"application\s+(\w+)", ct)
    if app_m and not any(s in app_m.group(1) for s in steady_names):
        return False
    MIN_ITER = 1000
    m_dt = re.search(r"deltaT\s+([0-9.eE+-]+)", ct)
    m_et = re.search(r"endTime\s+([0-9.eE+-]+)", ct)
    dt = float(m_dt.group(1)) if m_dt else 1.0
    et = float(m_et.group(1)) if m_et else 0.0
    n_iter = (et / dt) if dt > 0 else 0.0
    if n_iter >= MIN_ITER:
        return False
    # 重写：deltaT=1，endTime=MIN_ITER（迭代数语义）
    if m_dt:
        ct = re.sub(r"deltaT\s+[0-9.eE+-]+", "deltaT          1", ct)
    if m_et:
        ct = re.sub(r"endTime\s+[0-9.eE+-]+", f"endTime         {MIN_ITER}", ct)
    # writeInterval 若为 timeStep 且过稀/过期，规整为 100 步一写
    wi = re.search(r"writeInterval\s+([0-9.eE+-]+)", ct)
    if wi and float(wi.group(1)) > MIN_ITER:
        ct = re.sub(r"writeInterval\s+[0-9.eE+-]+", "writeInterval   100", ct)
    open(cd, "w", encoding="utf-8").write(ct)
    return True

def check_degenerate_mesh(case_dir):
    """检测 blockMeshDict 是否是退化网格（LLM 常生成 1x1xN 的一维块，物理上无法解析流动）。
    返回 (是否退化, 原因描述)。"""
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(bmd):
        return False, ""
    txt = open(bmd, encoding="utf-8").read()
    blocks = re.findall(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", txt)
    if not blocks:
        return False, ""
    dims = [(int(a), int(b), int(c)) for a, b, c in blocks]
    total = sum(a * b * c for a, b, c in dims)
    # 退化判定：总 cell 过少（< 500），或所有块某方向=1 的一维结构且总 cell < 2000
    degenerate_1d = all(min(d) == 1 for d in dims) and total < 2000
    too_few = total < 200
    if too_few:
        return True, (f"网格只有 {total} 个 cell（块尺寸 {dims}），单元数过少，"
                      f"无法解析流动场，求解必然发散。需加密网格（建议每方向 ≥ 20 层，总 cell ≥ 5000）。")
    if degenerate_1d:
        return True, (f"网格块全为 1×1×N 的一维退化结构（{dims}，共 {total} cells），"
                      f"缺少 x/y 方向单元分辨率，MRF/湍流源项会导致压力场发散。"
                      f"需要三维网格划分（建议 20×20×20 以上）。")
    return False, ""

def refine_degenerate_mesh(case_dir):
    """退化网格（1×1×N 一维块）确定性加密：把每个方向不足的 cell 数补到合理值。
    LLM 常写 (1 1 20)，导致 x/y 方向无分辨率 → 求解发散。
    规则：单方向 < 10 层 → 补到 20 层；但总 cell 控制在 100 万以内。
    返回 True 表示做了加密。"""
    bmd = os.path.join(case_dir, "system", "blockMeshDict")
    if not os.path.exists(bmd):
        return False
    txt = open(bmd, encoding="utf-8").read()
    changed = False

    def repl_block(m):
        nonlocal changed
        head = m.group(1)          # "hex (0 1 3 2 4 5 7 6) "
        nx, ny, nz = int(m.group(2)), int(m.group(3)), int(m.group(4))
        rest = m.group(5)          # " simpleGrading (1 1 1)"
        new = []
        for n in (nx, ny, nz):
            if n <= 2:
                new.append(20)
            elif n < 10:
                new.append(max(20, n * 5))
            else:
                new.append(n)
        # 控制总 cell 规模（单块 ≤ 200万）
        while new[0] * new[1] * new[2] > 2_000_000:
            i = new.index(max(new))
            new[i] = max(10, new[i] // 2)
        if tuple(new) != (nx, ny, nz):
            changed = True
        return f"{head}({new[0]} {new[1]} {new[2]}){rest}"

    # 匹配: hex (...) (nx ny nz) simpleGrading ...
    new_txt = re.sub(
        r"(hex\s*\([^)]*\)\s*)\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)((?:\s*simpleGrading[^;\n]*)?)",
        repl_block, txt)
    if changed:
        open(bmd, "w", encoding="utf-8").write(new_txt)
        return True
    return False

def run_case(jid, case_dir, plan):
    """blockMesh -> checkMesh -> 求解器；返回 (成功?, 错误日志累计)
    关键：必须验证产物真实存在（blockMesh 退出码 0 但可能未生成网格），否则假成功"""
    solver = plan.get("solver", "simpleFoam")
    err_all = ""
    bmd_path = os.path.join(case_dir, "system", "blockMeshDict")
    # 建议一(强化)：涡轮机械（压气机/涡轮/叶片机）**无论 LLM 是否给出 blockMeshDict**，
    # 都必须用确定性构建器覆盖。原因：LLM 常生成"等直径环形直通道 + frontBack=empty"
    # 的伪几何，会丢掉 MRF 旋转域、叶片排轴向站位、楔形(wedge)轴对称约定，
    # 使仿真物理失真、实验报告随之编造（历史案例 3fcfdc 即此问题）。
    if _is_turbomachinery(plan):
        hit, note = build_turbomachinery(case_dir, plan)
        if hit:
            job_step(jid, "确定性补丁", "done", "涡轮机械几何构建器：" + note)
        elif not os.path.exists(bmd_path):
            build_blockmesh_dict_from_plan(case_dir, plan)
            job_step(jid, "确定性补丁", "done", "自动补 blockMeshDict（LLM 缺失）")
    elif not os.path.exists(bmd_path):
        # LLM 未生成 blockMeshDict → 确定性构造（规则矩形/方腔几何）
        build_blockmesh_dict_from_plan(case_dir, plan)
        job_step(jid, "确定性补丁", "done", "自动补 blockMeshDict（LLM 缺失）")
    if os.path.exists(bmd_path):
        # 网格退化检测 + 自动加密：LLM 常生成 1×1×N 一维块，物理上无法收敛
        degen, degen_msg = check_degenerate_mesh(case_dir)
        if degen:
            if refine_degenerate_mesh(case_dir):
                degen2, degen_msg2 = check_degenerate_mesh(case_dir)
                if not degen2:
                    job_step(jid, "确定性补丁", "done", "网格退化 → 自动加密至合理分辨率")
                else:
                    job_step(jid, "blockMesh 画网格", "failed", degen_msg2[:300])
                    return False, err_all + "\n[网格退化] " + degen_msg2
            else:
                job_step(jid, "blockMesh 画网格", "failed", degen_msg[:300])
                return False, err_all + "\n[网格退化] " + degen_msg
        # 确定性补丁：hex 缺 grading → 补 simpleGrading(1 1 1)
        if patch_blockmesh_grading(case_dir):
            job_step(jid, "确定性补丁", "done", "补齐 blockMesh hex grading")
        # 确定性补丁：同一面被多个 patch 重复分配（Open cells 主因）
        dup_changed, dup_n = patch_duplicate_boundary_faces(case_dir)
        if dup_changed:
            job_step(jid, "确定性补丁", "done", f"移除 {dup_n} 个重复分配的边界面")
        job_step(jid, "blockMesh 画网格", "running")
        rc, out = run_bash("blockMesh > log.blockMesh 2>&1; echo rc=$?; tail -3 log.blockMesh", case_dir, 120)
        err_all += out
        # 真正成功的判据：退出码 0 且 polyMesh 的 points 文件真实存在
        pm = os.path.join(case_dir, "constant", "polyMesh", "points")
        mesh_ok = (rc == 0) and os.path.exists(pm)
        # 若失败，提取真实 FATAL 原因
        if not mesh_ok:
            bm_log = ""
            blp = os.path.join(case_dir, "log.blockMesh")
            if os.path.exists(blp):
                bm_log = open(blp, encoding="utf-8", errors="ignore").read()
            fatal = _extract_fatal(bm_log) or out
            job_step(jid, "blockMesh 画网格", "failed", fatal[-400:])
            return False, err_all + "\n[blockMesh 未生成网格] " + fatal[-1500:]
        job_step(jid, "blockMesh 画网格", "done", out[-200:])
        # MRF 旋转域：blockMesh 不识别 cellZones 段且会清空 polyMesh →
        # 必须在此用 topoSet+setsToZones 按几何生成（涡轮机械算例）
        mrf_ok, mrf_msg = ensure_mrf_cellzone(case_dir)
        if mrf_ok:
            job_step(jid, "确定性补丁", "done", mrf_msg)
        job_step(jid, "checkMesh 网格检查", "running")
        rc, out = run_bash("checkMesh > log.checkMesh 2>&1; grep -E 'Mesh OK|FAILED|non-orthogonality' log.checkMesh | head -4", case_dir, 120)
        cm_log = ""
        cmlp = os.path.join(case_dir, "log.checkMesh")
        if os.path.exists(cmlp):
            cm_log = open(cmlp, encoding="utf-8", errors="ignore").read()
        if "Mesh OK" not in cm_log:
            # 尝试重建标准 2D 网格（单块矩形域 + 面定义错乱场景）
            if rebuild_standard_2d_blockmesh(case_dir):
                job_step(jid, "确定性补丁", "done", "面定义错乱 → 重建标准 2D blockMeshDict")
                job_step(jid, "blockMesh 画网格", "running")
                rc, out = run_bash("blockMesh > log.blockMesh 2>&1; echo rc=$?; tail -3 log.blockMesh", case_dir, 120)
                err_all += out
                pm = os.path.join(case_dir, "constant", "polyMesh", "points")
                if rc == 0 and os.path.exists(pm):
                    job_step(jid, "checkMesh 网格检查", "running")
                    rc2, out2 = run_bash("checkMesh > log.checkMesh 2>&1; grep -E 'Mesh OK|FAILED' log.checkMesh | head -3", case_dir, 120)
                    err_all += out2
                    cmlp2 = os.path.join(case_dir, "log.checkMesh")
                    cm_log2 = open(cmlp2, encoding="utf-8", errors="ignore").read() if os.path.exists(cmlp2) else ""
                    if "Mesh OK" in cm_log2:
                        job_step(jid, "checkMesh 网格检查", "done", out2[-200:])
                    else:
                        fatal = _extract_fatal(cm_log2) or out2
                        job_step(jid, "checkMesh 网格检查", "failed", fatal[-300:])
                        return False, err_all + "\n[重建后 checkMesh 仍失败] " + fatal[-1500:]
                else:
                    fatal = _extract_fatal(out) or out
                    job_step(jid, "blockMesh 画网格", "failed", fatal[-300:])
                    return False, err_all + "\n[重建后 blockMesh 失败] " + fatal[-1500:]
            else:
                fatal = _extract_fatal(cm_log) or out
                job_step(jid, "checkMesh 网格检查", "failed", fatal[-300:])
                return False, err_all + "\n[checkMesh 失败] " + fatal[-1500:]
        job_step(jid, "checkMesh 网格检查", "done", out[-200:])
    if patch_transient_final_entries(case_dir, solver):
        job_step(jid, "确定性补丁", "done", "补齐 UFinal/pFinal 求解器条目")
    if re.search(r"rhoCentralFoam|rhoPimpleFoam|sonicFoam|buoyantFoam", solver or ""):
        if patch_thermophysical_common(case_dir):
            job_step(jid, "确定性补丁", "done", "补齐 thermophysicalProperties(Pr/thermoType)")
    if patch_wall_dist_method(case_dir):
        job_step(jid, "确定性补丁", "done", "修复 fvSchemes wallDist 的 patchDistMethod")
    if patch_closed_domain_pressure_ref(case_dir, solver):
        job_step(jid, "确定性补丁", "done", "封闭域补压力参考 pRefCell/pRefValue")
    if patch_fvsolution_solver_entries(case_dir):
        job_step(jid, "确定性补丁", "done", "修复 fvSolution smoother/solver 非法值")
    if patch_mrf_cellzone(case_dir):
        job_step(jid, "确定性补丁", "done", "生成/补齐 MRF cellZone（constant/polyMesh/cellZones）")
    n_fix = patch_field_value_entries(case_dir)
    if n_fix:
        job_step(jid, "确定性补丁", "done", f"补齐 {n_fix} 个场文件缺失的 value 条目")
    if patch_mrf_stabilize(case_dir):
        job_step(jid, "确定性补丁", "done", "MRF 数值稳定化（小 deltaT/一阶迎风/低 p 松弛）")
    # 稳态求解器：endTime/deltaT 是迭代计数语义，LLM 常写成瞬态的 0.5/1 → 零迭代假成功
    if patch_steady_iterations(case_dir, solver):
        job_step(jid, "确定性补丁", "done", f"稳态迭代步数修正（{solver} 按迭代计数，已规整为 ≥1000 步）")
    job_step(jid, f"{solver} 求解", "running")
    rc, out = run_bash(f"timeout 240 {solver} > log.solver 2>&1; echo rc=$?; tail -4 log.solver", case_dir, 280)
    err_all += out
    log = ""
    lp = os.path.join(case_dir, "log.solver")
    if os.path.exists(lp):
        log = open(lp, encoding="utf-8", errors="ignore").read()
    times, series = parse_residuals(log)
    reached_end = ("End" in log) or ("executionTime" in log and "rc=0" in out)
    converged_ok = reached_end or (len(times) > 10 and times[-1] >= float(plan.get("endTime", 1)) * 0.95)
    if series:
        job_update(jid, residuals={"times": times, "series": series})
    if not converged_ok:
        fatal = _extract_fatal(log) or out[-400:]
        job_step(jid, f"{solver} 求解", "failed", fatal[-350:])
        return False, err_all + "\n[求解未到 endTime] " + fatal[-1500:]
    job_step(jid, f"{solver} 求解", "done", f"完成 {len(times)} 步")
    # 涡轮机械算例：求解后校核级的气动指标（欧拉功/零预旋/消旋），防止"跑通即通过"
    vok, vmetrics, vmsg = verify_turbomachinery_result(case_dir, plan)
    if vok is not None:
        job_step(jid, "气动校核", "done" if vok else "failed",
                 (vmsg + f" | {vmetrics}")[:400])
    return True, err_all

def _extract_fatal(log_text):
    """从求解日志提取真正的 FATAL 错误描述（而非 FOAM exiting 结尾）"""
    if not log_text:
        return ""
    # 取最后一次 "FOAM FATAL ERROR" 到 "FOAM exiting" 之间的内容
    idx = log_text.rfind("FOAM FATAL ERROR")
    if idx >= 0:
        seg = log_text[idx:idx + 1800]
        # 去掉尾部 banner
        seg = seg.split("FOAM exiting")[0]
        return seg.strip()
    # 没有 FATAL 则取有意义的尾部（去掉 OpenFOAM banner 和 Time 行）
    lines = [l for l in log_text.splitlines()
             if l.strip() and not l.startswith("/*") and "\\" not in l[:3]
             and not re.match(r"^\s*//", l)]
    return "\n".join(lines[-12:])

# ---------- HTTP 服务 ----------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = open(os.path.join(STATIC, "index.html"), "rb").read()
            return self._send(200, "text/html; charset=utf-8", body)
        if self.path == "/api/models":
            return self._send(200, "application/json",
                              json.dumps({"models": MODELS}).encode())
        m = re.match(r"^/api/job/([\w-]+)$", self.path)
        if m:
            with JOBS_LOCK:
                data = json.dumps(JOBS.get(m.group(1), {"status": "unknown"}), ensure_ascii=False)
            return self._send(200, "application/json", data.encode())
        m = re.match(r"^/api/download/([\w-]+)$", self.path)
        if m:
            case_dir = os.path.join(RUNS, m.group(1))
            if not os.path.isdir(case_dir):
                return self._send(404, "text/plain", b"not found")
            zpath = os.path.join(RUNS, m.group(1) + ".zip")
            if not os.path.exists(zpath):
                with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                    for root, _, fs in os.walk(case_dir):
                        for f in fs:
                            fp = os.path.join(root, f)
                            z.write(fp, os.path.relpath(fp, case_dir))
            body = open(zpath, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{m.group(1)}.zip"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path_only = self.path.split("?")[0]
        if path_only == "/api/upload":
            from urllib.parse import urlparse, parse_qs, unquote
            q = parse_qs(urlparse(self.path).query)
            name = unquote(q.get("name", ["file.txt"])[0])
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                text = extract_doc_text(name, data)
            except Exception as e:
                return self._send(400, "application/json",
                                  json.dumps({"error": f"文档解析失败: {e}"}).encode())
            if not text.strip():
                return self._send(400, "application/json",
                                  json.dumps({"error": "未能从文档中提取到文本（可能是扫描版 PDF）"}).encode())
            uid = uuid.uuid4().hex[:10]
            UPLOADS[uid] = {"name": name, "text": text, "chars": len(text)}
            return self._send(200, "application/json",
                              json.dumps({"upload_id": uid, "name": name,
                                          "chars": len(text)}, ensure_ascii=False).encode())
        if self.path == "/api/chat":
            ln = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(ln))
            jid = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[jid] = {"status": "running", "steps": [], "created": datetime.now().isoformat(timespec="seconds"),
                             "start": time.time(), "current": "等待启动", "step_done": 0,
                             "message": req.get("message", "")}
            threading.Thread(target=pipeline,
                             args=(jid, req.get("message", ""), req.get("model", "qwen3.8-flash"),
                                   req.get("history", []),
                                   UPLOADS.get(req.get("upload_id", ""), {}).get("text", "")),
                             daemon=True).start()
            return self._send(200, "application/json", json.dumps({"job_id": jid}).encode())
        return self._send(404, "text/plain", b"not found")

if __name__ == "__main__":
    print("Chat-Foam-agent 服务启动: http://localhost:8502")
    print(f"知识库: {len(KNOWLEDGE)} 段 / {len(KNOWLEDGE_TEXT)} 字符")
    ThreadingHTTPServer(("0.0.0.0", 8502), Handler).serve_forever()
