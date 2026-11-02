#!/usr/bin/env python3
import sys, os
sys.path.insert(0, "src")
log = open("/tmp/chat_smoke_result.txt", "w", buffering=1)
def mark(m): log.write(m + "\n")
try:
    mark("start")
    import config; mark("config ok")
    import preprocess_OF_tutorial; mark("preprocess_OF_tutorial ok")
    import main_run_chatcfd; mark("main_run_chatcfd ok")
    import file_preparation; mark("file_preparation ok")
    from qa_modules import QA_NoContext_deepseek_R1; mark("qa_modules ok")
    from file_summary import get_case_content, generate_case_summary; mark("file_summary ok")
    import chatbot; mark("chatbot import ok")
    cb = chatbot.ChatBot()
    mark("ChatBot constructed")
    r = cb.get_response([{"role": "user", "content": "Reply with exactly: ChatCFD ready"}])
    mark("RESPONSE: " + str(r)[:150])
except BaseException as e:
    import traceback
    mark("EXC: " + type(e).__name__ + ": " + str(e))
    mark(traceback.format_exc())
finally:
    mark("end")
    log.close()
