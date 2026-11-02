#!/usr/bin/env python3
"""无头验证：chatbot.py 首页脚本能完整执行并渲染（不卡导入、无异常）。"""
import sys, os, time
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "src")

log = open("/tmp/st_verify_result.txt", "w", buffering=1)
def mark(m): log.write(m + "\n")

t0 = time.time()
try:
    from streamlit.testing.v1 import AppTest
    mark("AppTest imported")
    at = AppTest.from_file("src/chatbot.py", default_timeout=600)
    at.run()
    mark(f"script finished in {time.time()-t0:.1f}s")
    mark(f"exceptions: {[str(e.value) for e in at.exception]}")
    mark(f"titles: {[t.value for t in at.title]}")
    mark(f"headers: {[h.value for h in at.header][:3]}")
except BaseException as e:
    import traceback
    mark("EXC: " + type(e).__name__ + ": " + str(e))
    mark(traceback.format_exc())
finally:
    mark("end")
    log.close()
