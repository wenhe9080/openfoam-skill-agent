# ChatCFD 复现长期备忘

## 环境架构（Windows + WSL）
- 宿主 Windows，实际运行环境：WSL Ubuntu-22.04 + OpenFOAM v2406（/usr/lib/openfoam/openfoam2406）
- Python venv：/mnt/g/ChatCFD-main/chatcfd_venv（Python 3.11.17，PyFoam/faiss/sentence-transformers OK）
- LLM：DashScope compatible-mode（qwen-flash 替代 DeepSeek-R1/V3），key 在 inputs/chatcfd_config.json
- Git Bash 调 WSL 时必须加 MSYS_NO_PATHCONV=1，否则 /mnt/g 路径被转换
- WSL 无法访问 huggingface.co，但模型已缓存于 ~/.cache/huggingface/hub；驱动脚本里设 HF_HUB_OFFLINE=1

## 已完成
- NACA0012 案例（不可压 simpleFoam kOmegaSST）：test_chatcfd_smoke.py，ICOT 24 轮后 Success，端到端复现成功
- Nozzle 案例（可压缩 rhoCentralFoam RNGkEpsilon）：test_chatcfd_nozzle.py，端到端跑通，两次运行均 30 轮 ICOT 后 Failure（与论文自报 30% 成功率一致）
- 聊天界面：chatbot.py 自带 Streamlit 界面已跑通（run_chatcfd_web.sh 启动，端口 8501）；ChatBot 自由对话/JSON转换两通道已补 enable_thinking=False；test_chat_channel.py 为聊天链路冒烟测试

## 复现中修复的上游代码 bug（src/）
1. config.py：llm_config dataclass 缺 DEEPSEEK_V3/R1_MODEL_NAME 字段 → JSON 配置被过滤，环境变量回退 DeepSeek 默认模型名 → 404。已补字段。
2. file_preparation.py case_required_files：LLM 编造参考案例名重试 3 次后 response.lower()=="none" 是无操作语句 → file_structure 未赋值 UnboundLocalError。改为 response="none" 走回退分支。
3. file_corrector.py extract_content_from_response：LLM 返回不带 ``` 围栏的裸 JSON 时被双重编码返回 str → 上层 bc_info[i][j] 崩溃 → generate_initial_files 无限重试。已加二次解码。
4. file_preparation.py generate_initial_files：BC 提取加了结构校验+局部重试（3 次）。
5. qa_modules.py：qwen-flash 混合思考模型默认 thinking，40k prompt 流式思考 15-20 分钟 → 三个调用点加 extra_body={"enable_thinking": False}，速度提升 10 倍+。

## 关键经验
- DashScope 偶发单请求挂起 >10min，需耐心或加超时
- run_chatcfd/sample_nozzle/sample_SERN_NPR3_rhoCentralFoam_0/ 下有 qa_logs.jsonl（token 统计）、error_history.txt（调试轨迹）
- 首次 nozzle 运行统计：202 次 LLM 调用（V3 146 次 117k tok，R1 56 次 429k tok）
