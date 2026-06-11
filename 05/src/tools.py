import subprocess
import os
from .security import security_check, is_allowed_path, request_approval, resolve_path

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command for reading files, listing directories, compiling, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file (supports multi-line, safe cross-platform). Use this instead of shell redirection for creating files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "File content"}
            },
            "required": ["path", "content"]
        }
    },
]


def execute_tool(name, tool_input):
    print(f"\n=== 工具執行 ===")
    print(f"工具：{name}")
    print(f"輸入：{tool_input}")

    if name == "run_command":
        command = tool_input.get("command", "")

        allowed, reason = security_check(command)
        if not allowed:
            msg = f"[安全攔截] 指令被拒絕：{reason}"
            print(msg)
            print(f"=== 結束 ===\n")
            return msg

        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=30,
                cwd=os.getcwd()
            )
            output = result.stdout + result.stderr
            print(f"結果：{output}")
            print(f"=== 結束 ===\n")
            return output if output else "（無輸出）"
        except Exception as e:
            print(f"錯誤：{e}")
            print(f"=== 結束 ===\n")
            return f"錯誤：{e}"

    if name == "write_file":
        path = tool_input.get("path", "")
        content = tool_input.get("content", "")

        if not is_allowed_path(path):
            print(f"\n🔒 [安全攔截] 寫入目標不在安全區域內")
            print(f"   路徑: {path}")
            approved = request_approval([path])
            if not approved:
                msg = "[安全攔截] 寫入被拒絕"
                print(msg)
                print(f"=== 結束 ===\n")
                return msg

        try:
            abs_path = resolve_path(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            msg = f"✅ 已寫入 {abs_path}（{len(content)} bytes）"
            print(msg)
            print(f"=== 結束 ===\n")
            return msg
        except Exception as e:
            print(f"錯誤：{e}")
            print(f"=== 結束 ===\n")
            return f"寫入失敗：{e}"

    print(f"=== 結束 ===\n")
    return f"未知工具：{name}"
