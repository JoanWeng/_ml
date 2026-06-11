> 本專案由opencode製作

# agent0.py — AI Agent 安全控管版

## 概述

這是一個基於 Gemini API 的 AI 代理程式，能理解使用者的自然語言指令，並自動執行工具來完成任務。本版本著重於**安全控管**，防止 AI 未經允許存取程式資料夾以外的檔案。

## 檔案結構

```
05/
├── agent0.py          ← 入口點（3 行，呼叫 src.agent.main()）
├── src/
│   ├── config.py      ← 設定常數（API Key、模型、安全區域）
│   ├── security.py    ← 安全審查（路徑提取、攔截、核可）
│   ├── gemini.py      ← Gemini API 封裝
│   ├── memory.py      ← 記憶管理（對話歷史、關鍵資訊）
│   ├── tools.py       ← 工具定義與執行（run_command + write_file）
│   └── agent.py       ← 主邏輯 + main()
├── _doc/
│   ├── 對話記錄.md
│   └── 程式架構與運行.md
└── README.md           ← 本文件
```

## 運行方式

### 前置需求

- Python 3.10+
- Gemini API Key（[申請](https://aistudio.google.com/apikey)）
- `google-genai` 套件

```bash
pip install google-genai
```

### 設定

API Key 透過環境變數設定（不寫死在程式碼中）：

```bash
set GEMINI_API_KEY=你的API_KEY       # CMD
$env:GEMINI_API_KEY="你的API_KEY"    # PowerShell
```

### 啟動

```bash
python agent0.py
```

### 指令

| 指令 | 功能 |
|------|------|
| 直接輸入 | 與 AI 對話 |
| `/quit` | 離開 |
| `/memory` | 顯示 AI 記住的關鍵資訊 |
| `/approved` | 顯示本次已核可的外部路徑 |

---

## 原理

### 整體流程

```
使用者輸入 → 組合對話上下文 → 呼叫 Gemini API
                                   ↓
                          AI 回覆（可能含 <tool> 標籤）
                                   ↓
                       解析 <tool> JSON → 執行工具
                          ├─ run_command → 安全審查（路徑檢查）
                          │                  ├─ 通過 ✓ → 執行 shell 指令
                          │                  └─ 拒絕 ✗ → 回傳錯誤
                          └─ write_file  → 安全審查（路徑檢查）
                                             ├─ 通過 ✓ → Python 原生寫檔
                                             └─ 拒絕 ✗ → 回傳錯誤
                                   ↓
                       結果送回 AI → 再次呼叫
                                   ↓
                        無 <tool> → 輸出最終回覆
```

### 核心模組

| 模組（檔案） | 功能 |
|-------------|------|
| `src/config.py` | 設定常數（`WORKSPACE`、`MODEL`、`PROGRAM_DIR`） |
| `src/security.py` | 安全審查（`security_check()`、`extract_paths_from_command()`、`is_allowed_path()`、`request_approval()`） |
| `src/gemini.py` | Gemini API 封裝（`call_gemini()`） |
| `src/memory.py` | 記憶管理（`build_context()`、`update_memory()`、`extract_key_info()`） |
| `src/tools.py` | 工具定義與執行（`execute_tool()`：`run_command` + `write_file`） |
| `src/agent.py` | 主邏輯 + SYSTEM_PROMPT + `main()` |

### 記憶機制

- **對話歷史**：保留最近 `MAX_TURNS * 4` 則訊息（預設 20 則），每次呼叫時一併送入 AI
- **關鍵資訊**：AI 每次工具執行後，會自動判斷是否有需要長期記住的資訊（透過另一個 Gemini API 呼叫萃取）

---

## 安全控管

### 設計原則

1. **預設拒絕**：程式資料夾以外的檔案，一律不可直接存取
2. **最小權限**：AI 只能透過 `run_command`（shell）或 `write_file`（Python 原生）操作檔案
3. **人機協作**：外部檔案的存取需要使用者當面核可

### 安全區域

```
程式資料夾（agent0.py 所在目錄）→ 自動允許，無須詢問
外部路徑                         → 攔截，詢問使用者
```

### 攔截流程

```
AI 產生指令：cat /etc/passwd
       ↓
extract_paths_from_command()
  提取路徑：["/etc/passwd"]
       ↓
is_allowed_path("/etc/passwd")
  → 不在 PROGRAM_DIR 內
  → 不在 approved_paths 中
  → 回傳 False
       ↓
security_check() 偵測到外部路徑
  → 輸出 🔒 [安全攔截] 訊息
  → 呼叫 request_approval()
       ↓
詢問使用者：是否允許存取？[y/n/always]
  ├─ y      → 本次允許，執行指令
  ├─ n      → 拒絕，回傳拒絕訊息
  └─ always → 記住此路徑（本次執行期間不再詢問），執行指令
```

### 路徑偵測

`extract_paths_from_command()` 使用正則表達式從指令中偵測以下模式：

| 類別 | 範例 |
|------|------|
| 讀取指令 | `cat file`, `less file`, `grep pattern file`, `ls path` |
| 寫入指令 | `echo > file`, `printf ... > file` |
| 複製/移動/刪除 | `cp a b`, `mv a b`, `rm file` |
| 目錄操作 | `cd path`, `mkdir path` |
| 相對跳脫 | `../file`, `../../dir` |
| 絕對路徑 | `/etc/passwd`, `/home/user` |
| 家目錄 | `~/file` |

---

## 本版改動

### 相較於原始版本（v2）的變更

1. **重新定義安全區域**
   - 新增 `PROGRAM_DIR` = `agent0.py` 所在目錄
   - 只有此目錄（含子目錄）為自動允許的安全區域
   - `WORKSPACE` (~/.agent0) 不再自動允許（外部目錄）

2. **強化路徑偵測**
   - 新增 `cd` 指令路徑提取
   - 新增 `../` 相對跳脫路徑提取
   - 新增絕對路徑 `/xxx` 提取
   - 新增家目錄 `~/xxx` 提取

3. **完善安全攔截流程**
   - `security_check()` 在執行指令前先分析所有路徑
   - 外部路徑觸發 `request_approval()` 與使用者互動
   - 支援 `always` 選項，本次執行期間不再重複詢問

4. **繁體中文界面**
   - 所有使用者端訊息改為正體中文
   - System prompt 改為中文，確保 AI 理解指令格式

### 相較於原始版本（v3）的變更（本版）

1. **新增 `write_file` 工具**
   - Python 原生寫檔，繞過 shell，解決 Windows cmd.exe 不支援 `<<` heredoc 的問題
   - 自動安全審查，支援跨平台
   - System prompt 引導 AI 優先使用 `write_file` 而非 shell 寫檔

2. **強化路徑提取**
   - `extract_paths_from_command()` 在匹配前先移除引號內內容，避免 printf/echo 的 format string 被誤認為檔案路徑（如 `\n\nvoid`）

3. **模組拆分**
   - 將原本單一 `agent0.py` 拆分為 `src/` 套件（6 個模組），提升可維護性
   - 使用相對匯入（`from .config import ...`）避免路徑衝突

### 參考來源

本實作參考 [ccc114b/cccocw 的 v3-agent-secure](https://github.com/ccc114b/cccocw/tree/main/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92/07b-%E4%BB%A3%E7%90%86%E4%BA%BA/agent0/v3-agent-secure)，將其中的安全控管概念整合至 Gemini-based agent。

---

## 授權

MIT License
