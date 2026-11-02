# ChatCFD 环境变量配置说明

本项目的 API 密钥**不存放在代码或配置文件中**，而是通过系统环境变量提供。

## 必需的环境变量

| 变量名 | 说明 | 示例值 |
|---|---|---|
| `DEEPSEEK_V3_KEY` | DashScope（通义千问）API Key | `sk-xxxxxxxx...` |
| `DEEPSEEK_R1_KEY` | 同上（可与 V3 相同） | `sk-xxxxxxxx...` |
| `DEEPSEEK_V3_BASE_URL` | API 基础地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DEEPSEEK_R1_BASE_URL` | 同上 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `DEEPSEEK_V3_MODEL_NAME` | 模型名 | `qwen-flash` |
| `DEEPSEEK_R1_MODEL_NAME` | 模型名 | `qwen-flash` |

## 配置方式

### Windows（PowerShell，用户级）
```powershell
[Environment]::SetEnvironmentVariable('DEEPSEEK_V3_KEY', 'sk-你的密钥', 'User')
[Environment]::SetEnvironmentVariable('DEEPSEEK_R1_KEY', 'sk-你的密钥', 'User')
[Environment]::SetEnvironmentVariable('DEEPSEEK_V3_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'User')
[Environment]::SetEnvironmentVariable('DEEPSEEK_R1_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'User')
[Environment]::SetEnvironmentVariable('DEEPSEEK_V3_MODEL_NAME', 'qwen-flash', 'User')
[Environment]::SetEnvironmentVariable('DEEPSEEK_R1_MODEL_NAME', 'qwen-flash', 'User')
```
设置后**重启终端**（或重新登录）使其生效。

### Linux / WSL
```bash
export DEEPSEEK_V3_KEY="sk-你的密钥"
export DEEPSEEK_R1_KEY="sk-你的密钥"
export DEEPSEEK_V3_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DEEPSEEK_R1_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export DEEPSEEK_V3_MODEL_NAME="qwen-flash"
export DEEPSEEK_R1_MODEL_NAME="qwen-flash"
```
建议写入 `~/.bashrc` 或 `~/.profile`。

## 代码中的读取方式

- `src/config.py`：先读 `inputs/chatcfd_config.json`，若 KEY 为空则回退读环境变量
- `foam_agent/app.py`：`CFG.get("DEEPSEEK_V3_KEY") or os.environ.get("DEEPSEEK_V3_KEY")`
- `src/chatbot.py`、`src/pdf_chunk_ask_question.py`：直接 `os.environ.get("DEEPSEEK_R1_KEY")`

## 安全提示

- **不要**把密钥写进任何 `.py` / `.json` / `.md` 文件
- **不要**提交 `.env`、`*api_key*`、`temp/backup_keys/` 到 Git
- 密钥泄露后请立即到 DashScope 控制台**重置**
