#!/usr/bin/env python3
"""快速验证 DashScope / Qwen 模型 API 可用性

密钥来源（安全，不硬编码）：
  1. 环境变量 DEEPSEEK_V3_KEY
  2. 回退读取 inputs/chatcfd_config.json（该文件不存明文密钥，仅作兼容）
"""
import json, os, sys
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_V3_KEY", "")
if not API_KEY:
    # 兼容旧配置：尝试从 chatcfd_config.json 读（现在该文件应为空，仅作 fallback）
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "inputs", "chatcfd_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            API_KEY = json.load(f).get("DEEPSEEK_V3_KEY", "")
    except Exception:
        API_KEY = ""
BASE_URL = os.environ.get("DEEPSEEK_V3_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("DEEPSEEK_V3_MODEL_NAME", "qwen-flash")

if not API_KEY:
    print("错误：未找到 DEEPSEEK_V3_KEY。请设置系统环境变量 DEEPSEEK_V3_KEY。")
    sys.exit(2)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

messages = [
    {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
    {"role": "user", "content": "What is CFD? Reply in one sentence."}
]

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=128
    )
    content = resp.choices[0].message.content
    print(f"SUCCESS — model={MODEL}")
    print(f"response: {content}")
    print(f"usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
