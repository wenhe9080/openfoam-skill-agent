#!/usr/bin/env bash
# ChatCFD 聊天界面启动脚本（在 WSL 内运行）
# 用法: bash run_chatcfd_web.sh   然后在 Windows 浏览器打开 http://localhost:8501
cd /mnt/g/ChatCFD-main
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec ./chatcfd_venv/bin/python -m streamlit run src/chatbot.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
