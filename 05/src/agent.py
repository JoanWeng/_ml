import os
import re
import json
from .config import MODEL, WORKSPACE, PROGRAM_DIR
from .security import approved_paths
from .gemini import call_gemini
from .memory import conversation_history, key_info, build_context, update_memory, extract_key_info
from .tools import execute_tool

SYSTEM_PROMPT = """你是 Jarvis，一個有用的 AI 助理。

你有工具可以使用：
- run_command：執行 shell 指令（讀檔、編譯、列出目錄等）
- write_file：將內容寫入檔案（支援多行，安全跨平台）

寫入檔案時請「務必」使用 write_file 工具，不要用 shell 指令寫檔（因 cmd.exe 不支援 << heredoc 語法）。

當你需要使用工具時，輸出以下格式：
<tool>
{"name": "tool_name", "input": {"key": "value"}}
</tool>

否則直接回覆即可。"""


def main():
    os.makedirs(WORKSPACE, exist_ok=True)

    print(f"Agent0 - {MODEL}（記憶 + 安全控管）")
    print(f"程式資料夾（安全區域）：{PROGRAM_DIR}")
    print(f"工作區：{WORKSPACE}")
    print("指令：/quit、/memory（顯示關鍵資訊）、/approved（顯示已核可路徑）")
    print("─────────────────────────────────────────────")
    print("🔒 安全控管已啟用：只能直接存取程式資料夾內的檔案")
    print("   存取外部檔案時會先攔截並詢問您是否允許")
    print("─────────────────────────────────────────────\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("再見！")
            break
        if user_input.lower() == "/memory":
            print(f"關鍵資訊：{key_info}")
            continue
        if user_input.lower() == "/approved":
            if approved_paths:
                print("本輪已核可的外部路徑：")
                for p in sorted(approved_paths):
                    print(f"  ✅ {p}")
            else:
                print("尚無已核可的外部路徑。")
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

            follow_up_prompt = f"""先前的對話：{context}

使用者：{user_input}
助理之前的回覆：{current_response}
工具輸出：
{chr(10).join(all_tool_outputs)}

如果你還需要更多工具，請輸出 <tool>。否則，直接給使用者最終回覆："""
            current_response = call_gemini(follow_up_prompt, SYSTEM_PROMPT)

        response = current_response

        print(f"\n🤖 {response}\n")

        update_memory(user_input, response, tool_result)
        if tool_result:
            extract_key_info(user_input, response)
