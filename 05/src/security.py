import os
import re
from .config import PROGRAM_DIR

approved_paths: set[str] = set()

FILE_ACCESS_PATTERNS = [
    r'\bcat\b\s+([^\s|&;<>]+)',
    r'\bless\b\s+([^\s|&;<>]+)',
    r'\bmore\b\s+([^\s|&;<>]+)',
    r'\bhead\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\btail\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bwc\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bgrep\b\s+(?:-\S+\s+)*\S+\s+([^\s|&;<>]+)',
    r'\bstat\b\s+([^\s|&;<>]+)',
    r'\bls\b\s+([^\s|&;<>]+)',
    r'\bcd\b\s+([^\s|&;<>]+)',
    r'>\s*([^\s|&;<>]+)',
    r'>>\s*([^\s|&;<>]+)',
    r'\bcp\b\s+(?:-\S+\s+)*([^\s|&;<>]+)\s+([^\s|&;<>]+)',
    r'\bmv\b\s+(?:-\S+\s+)*([^\s|&;<>]+)\s+([^\s|&;<>]+)',
    r'\brm\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\btouch\b\s+([^\s|&;<>]+)',
    r'\bmkdir\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bchmod\b\s+\S+\s+([^\s|&;<>]+)',
    r'\bchown\b\s+\S+\s+([^\s|&;<>]+)',
    r'\bnano\b\s+([^\s|&;<>]+)',
    r'\bvim\b\s+([^\s|&;<>]+)',
    r'\bvi\b\s+([^\s|&;<>]+)',
    r'\bprintf\b\s+.+?\s+([^\s|&;<>]+)',
    r'\becho\b\s+.+?>\s*([^\s|&;<>]+)',
    r'\btee\b\s+(?:-\S+\s+)*([^\s|&;<>]+)',
    r'\bfind\b\s+([^\s|&;<>-][^\s|&;<>]*)',
    r'(?:^|\s)(\.\.[/\\][^\s|&;<>]*)',
    r'(?:^|\s)(/[^\s|&;<>]+)',
    r'(?:^|\s)(~[^\s|&;<>]*)',
]


def extract_paths_from_command(command: str) -> list[str]:
    cleaned = re.sub(r"'[^']*'", "", command)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)
    paths = []
    for pattern in FILE_ACCESS_PATTERNS:
        for match in re.finditer(pattern, cleaned):
            for group in match.groups():
                if group:
                    p = group.strip().strip('"\'')
                    if p and not p.startswith('-') and not p.startswith('$'):
                        paths.append(p)
    return list(set(paths))


def resolve_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return os.path.normpath(path)


def is_allowed_path(path: str) -> bool:
    abs_path = resolve_path(path)
    abs_program_dir = os.path.normpath(PROGRAM_DIR)

    if abs_path.startswith(abs_program_dir + os.sep) or abs_path == abs_program_dir:
        return True

    if abs_path in approved_paths:
        return True

    return False


def request_approval(paths: list[str]) -> bool:
    abs_paths = [resolve_path(p) for p in paths]

    print("\n" + "="*60)
    print("  ⚠️  安全攔截：偵測到存取程式資料夾以外的檔案")
    print("="*60)
    for ap in abs_paths:
        print(f"     📁 {ap}")
    print()
    print(f"  安全區域（自動允許）：{os.path.normpath(PROGRAM_DIR)}")
    print()

    while True:
        try:
            answer = input("  是否允許存取？[y/n/always] (y=本次允許, n=拒絕, always=永久允許): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  已拒絕。")
            return False

        if answer in ('y', 'yes', '是'):
            print("  ✅ 已允許（本次）。\n")
            return True
        elif answer in ('always', 'a'):
            for ap in abs_paths:
                approved_paths.add(ap)
            print("  ✅ 已允許並記住（本次執行期間不再詢問）。\n")
            return True
        elif answer in ('n', 'no', '否'):
            print("  ❌ 已拒絕存取。\n")
            return False
        else:
            print("  請輸入 y、n 或 always。")


def security_check(command: str) -> tuple[bool, str]:
    paths = extract_paths_from_command(command)

    if not paths:
        return True, "no file paths detected"

    outside_paths = [p for p in paths if not is_allowed_path(p)]

    if not outside_paths:
        return True, "all paths within allowed zones"

    print(f"\n🔒 [安全攔截] 指令嘗試存取程式資料夾以外的檔案")
    print(f"   指令: {command}")
    approved = request_approval(outside_paths)
    if approved:
        return True, "user approved"
    else:
        return False, f"access denied to: {', '.join(resolve_path(p) for p in outside_paths)}"
