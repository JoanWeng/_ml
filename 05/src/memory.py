import json
import re
from .config import MAX_TURNS
from .gemini import call_gemini

conversation_history: list[str] = []
key_info: list[str] = []


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
    extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，輸出 JSON 陣列（最多 2 項）。如果沒有，輸出空的陣列 []。

對話：
使用者：{user_input}
助理：{assistant_response}

要記憶的關鍵資訊："""

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
