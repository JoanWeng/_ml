import os

WORKSPACE = os.path.expanduser("~/.agent0")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("❌ 錯誤：環境變數 GEMINI_API_KEY 未設定或為空值")
    print("   請先設定環境變數後重試，例如：")
    print("   set GEMINI_API_KEY=你的API_KEY  (CMD)")
    print("   $env:GEMINI_API_KEY=\"你的API_KEY\"  (PowerShell)")
    exit(1)

MODEL = "gemini-3.1-flash-lite-preview"
MAX_TURNS = 5
PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
