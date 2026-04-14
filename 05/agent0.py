#!/usr/bin/env python3
# agent0.py - AI Agent with memory and tool feedback
# Run: python agent0.py

import subprocess
import json
import os
import re
import shlex
from google import genai

# ─── Configuration ───

WORKSPACE = os.path.expanduser("~/.agent0")
GEMINI_API_KEY = "貼上你的API_KEY"   # ← 換成你的 Gemini API Key
MODEL = "gemini-2.0-flash"
MAX_TURNS = 5

# ─── Gemini Client ───

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Security ───

# 永久核可的路徑集合（本次執行期間記住使用者核可過的路徑）
approved_paths: set[str] = set()

# 已知的 shell 指令關鍵字（會帶路徑參數的），用於輔助路徑提取
FILE_ACCESS_PATTERNS = [
    # 讀取類
    r'\bcat\b\s+([^\s|&;<>]+)',
    r'\bless\b\s+([^\s|&;<>]+)',
    r'\bmore\b\s+([^\s|&;<>]+)',
    r'\bhead\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\btail\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bwc\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bgrep\b\s+(?:-\S+\s+)*\S+\s+([^\s|&;<>]+)',
    r'\bstat\b\s+([^\s|&;<>]+)',
    r'\bls\b\s+([^\s|&;<>]+)',
    # 寫入類
    r'>\s*([^\s|&;<>]+)',
    r'>>\s*([^\s|&;<>]+)',
    # 複製/移動/刪除
    r'\bcp\b\s+(?:-\S+\s+)*([^\s|&;<>]+)\s+([^\s|&;<>]+)',
    r'\bmv\b\s+(?:-\S+\s+)*([^\s|&;<>]+)\s+([^\s|&;<>]+)',
    r'\brm\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\btouch\b\s+([^\s|&;<>]+)',
    r'\bmkdir\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bchmod\b\s+\S+\s+([^\s|&;<>]+)',
    r'\bchown\b\s+\S+\s+([^\s|&;<>]+)',
    # 編輯器
    r'\bnano\b\s+([^\s|&;<>]+)',
    r'\bvim\b\s+([^\s|&;<>]+)',
    r'\bvi\b\s+([^\s|&;<>]+)',
    # printf/echo 寫檔
    r'\bprintf\b\s+.+?\s+([^\s|&;<>]+)',
    r'\becho\b\s+.+?>\s*([^\s|&;<>]+)',
    # tee
    r'\btee\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    # find
    r'\bfind\b\s+([^\s|&;<>-][^\s|&;<>]*)',
]


def extract_paths_from_command(command: str) -> list[str]:
    """從 shell 指令中提取所有檔案路徑"""
    paths = []
    for pattern in FILE_ACCESS_PATTERNS:
        for match in re.finditer(pattern, command):
            for group in match.groups():
                if group:
                    # 去除引號
                    p = group.strip().strip('"\'')
                    # 排除 flag、變數、空字串
                    if p and not p.startswith('-') and not p.startswith('$'):
                        paths.append(p)
    return list(set(paths))


def resolve_path(path: str) -> str:
    """解析成絕對路徑"""
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return os.path.normpath(path)


def is_allowed_path(path: str) -> bool:
    """
    判斷路徑是否在允許範圍內：
    - WORKSPACE 目錄（含子目錄）
    - 當前工作目錄（含子目錄）
    - 使用者本次已核可的路徑
    """
    abs_path = resolve_path(path)
    abs_workspace = os.path.normpath(WORKSPACE)
    abs_cwd = os.path.normpath(os.getcwd())

    # 在 WORKSPACE 內
    if abs_path.startswith(abs_workspace + os.sep) or abs_path == abs_workspace:
        return True

    # 在當前工作目錄內
    if abs_path.startswith(abs_cwd + os.sep) or abs_path == abs_cwd:
        return True

    # 使用者本次已核可
    if abs_path in approved_paths:
        return True

    return False


def request_approval(paths: list[str]) -> bool:
    """
    詢問使用者是否核可存取這些路徑外的檔案。
    回傳 True 表示核可，False 表示拒絕。
    """
    abs_paths = [resolve_path(p) for p in paths]

    print("\n" + "="*50)
    print("⚠️  安全警告：偵測到存取受保護區域外的檔案")
    print("="*50)
    for ap in abs_paths:
        print(f"   📁 {ap}")
    print()
    print("允許的安全區域：")
    print(f"   ✅ {os.path.normpath(WORKSPACE)}  (WORKSPACE)")
    print(f"   ✅ {os.path.normpath(os.getcwd())}  (當前目錄)")
    print()

    while True:
        try:
            answer = input("是否核可存取上述路徑？[y/n/always] (y=本次允許, n=拒絕, always=永久允許): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已拒絕。")
            return False

        if answer in ('y', 'yes', '是'):
            print("✅ 已核可（本次）。\n")
            return True
        elif answer in ('always', 'a'):
            for ap in abs_paths:
                approved_paths.add(ap)
            print("✅ 已核可並記住（本次執行期間）。\n")
            return True
        elif answer in ('n', 'no', '否'):
            print("❌ 已拒絕存取。\n")
            return False
        else:
            print("請輸入 y、n 或 always。")


def security_check(command: str) -> tuple[bool, str]:
    """
    執行指令前的安全審查。
    回傳 (allowed: bool, reason: str)
    """
    paths = extract_paths_from_command(command)

    if not paths:
        # 沒有偵測到明確路徑，預設允許（純計算/網路指令等）
        return True, "no file paths detected"

    outside_paths = [p for p in paths if not is_allowed_path(p)]

    if not outside_paths:
        return True, "all paths within allowed zones"

    # 有路徑在安全區域外，需要詢問
    approved = request_approval(outside_paths)
    if approved:
        return True, "user approved"
    else:
        return False, f"access denied to: {', '.join(resolve_path(p) for p in outside_paths)}"


# ─── Memory ───

conversation_history = []
key_info = []

# ─── Tools ───

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command (use 'cat file' to read, 'echo content > file' or 'printf ... > file' to write)",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"}
            },
            "required": ["command"]
        }
    },
]

# ─── Gemini API ───

def call_gemini(prompt: str, system: str = "") -> str:
    """Call Gemini API"""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = _gemini_client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
    )
    return response.text.strip()

# ─── Tool Execution ───

def execute_tool(name, tool_input):
    print(f"\n=== TOOL EXECUTE ===")
    print(f"Tool: {name}")
    print(f"Input: {tool_input}")
    
    if name == "run_command":
        command = tool_input.get("command", "")

        # ── 安全審查 ──
        allowed, reason = security_check(command)
        if not allowed:
            msg = f"[安全攔截] 指令被拒絕：{reason}"
            print(msg)
            print(f"=== END ===\n")
            return msg

        try:
            result = subprocess.run(
                command, shell=True, 
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd()
            )
            output = result.stdout + result.stderr
            print(f"Result: {output}")
            print(f"=== END ===\n")
            return output if output else "(no output)"
        except Exception as e:
            print(f"Error: {e}")
            print(f"=== END ===\n")
            return f"Error: {e}"
    
    print(f"=== END ===\n")
    return f"Unknown tool: {name}"

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        context_parts.append("Key information:\n" + "\n".join(f"- {k}" for k in key_info))
    if conversation_history:
        context_parts.append("Recent conversation:\n" + "\n".join(conversation_history[-MAX_TURNS*2:]))
    return "\n\n".join(context_parts)

def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
    if tool_result:
        conversation_history.append(f"Tool result: {tool_result[:500]}")
    
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)

def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""Based on this conversation, should any key information be remembered long-term?
If yes, output a JSON list of key points (max 2). If no, output an empty list [].

Conversation:
User: {user_input}
Assistant: {assistant_response}

Key information to remember:"""
    
    try:
        result = call_gemini(extract_prompt, "")
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            items = json.loads(match.group(0))
            for item in items:
                if item not in key_info:
                    key_info.append(item)
    except:
        pass

# ─── Agent ───

SYSTEM_PROMPT = """You are Jarvis, a helpful AI assistant.
You have tools - use them to help the user.

Available tools:
- run_command: Run a shell command. Use 'cat file' to read files, 'echo content > file' or 'printf' to write files.

When you need to use a tool, output in this format:
<tool>
{"name": "tool_name", "input": {"key": "value"}}
</tool>

Otherwise, just respond directly."""

def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    
    print(f"Agent0 - {MODEL} (with memory + security)")
    print(f"Workspace: {WORKSPACE}")
    print(f"Safe zones: {WORKSPACE}  |  {os.getcwd()}")
    print("Commands: /quit, /memory (show key info), /approved (show approved paths)\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            break
        if user_input.lower() == "/memory":
            print(f"Key info: {key_info}")
            continue
        if user_input.lower() == "/approved":
            if approved_paths:
                print("本次已核可的額外路徑：")
                for p in approved_paths:
                    print(f"  - {p}")
            else:
                print("（尚無額外核可路徑）")
            continue
        
        context = build_context()
        full_prompt = f"{context}\n\nUser: {user_input}" if context else f"User: {user_input}"
        
        response = call_gemini(full_prompt, SYSTEM_PROMPT)
        
        tool_result = None
        current_response = response
        
        while True:
            tool_matches = re.findall(r'<tool>(.+?)</tool>', current_response, re.DOTALL)
            if not tool_matches:
                break
            
            all_tool_outputs = []
            for tool_match in tool_matches:
                try:
                    tool_json = tool_match.strip()
                    tool_data = json.loads(tool_json)
                    tool_name = tool_data.get("name")
                    tool_input = tool_data.get("input", {})
                    
                    tool_output = execute_tool(tool_name, tool_input)
                    all_tool_outputs.append(f"[{tool_name}]: {tool_output}")
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Raw tool data: {tool_match}")
                except Exception as e:
                    print(f"Tool error: {e}")
            
            tool_result = (tool_result or "") + "\n" + "\n".join(all_tool_outputs)
            
            follow_up_prompt = f"""Previous context: {context}

User: {user_input}
Previous assistant responses: {current_response}
Tool outputs:
{chr(10).join(all_tool_outputs)}

If you need more tools, output them. Otherwise, provide your final response to the user:"""
            current_response = call_gemini(follow_up_prompt, SYSTEM_PROMPT)
        
        response = current_response
        
        print(f"\n🤖 {response}\n")
        
        update_memory(user_input, response, tool_result)
        if tool_result:
            extract_key_info(user_input, response)

if __name__ == "__main__":
    main()