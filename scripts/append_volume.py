"""
PHENO TYPE -- Volume 4+5 Append Script
========================================
Appends new samples from WinMET_volume_4 and WinMET_volume_5.
- Skips AgentTesla and Redline (already at 500 cap)
- Skips any sha256 already in final_dna_v2.csv
- Takes ALL available Formbook, Lokibot, njRAT from volumes 4+5
- Uses existing vocab from final_dna_v2_vocab.json (no rebuild)
- Appends to final_dna_v2.csv (no overwrite of existing rows)
"""

import json, csv, pathlib, sys
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 1200
PAD_TOKEN       = 0
UNKNOWN_TOKEN   = 1

BASE = pathlib.Path(__file__).parent

WINMET_VOLUMES_NEW = [BASE / "WinMET_volume_4", BASE / "WinMET_volume_5"]
LABEL_DIR        = BASE / "Winmet  DataSet Stuff"
CAPE_LABELS_FILE = LABEL_DIR / "cape_report_to_label_mapping.json"
AVC_LABELS_FILE  = LABEL_DIR / "avclass_report_to_label_mapping.json"

OUTPUT_CSV   = BASE / "final_dna_v2.csv"
OUTPUT_VOCAB = BASE / "final_dna_v2_vocab.json"

# Families to skip (already at cap)
SKIP_FAMILIES = {"AgentTesla", "Redline"}

FAMILY_MAP = {
    "Agenttesla": "AgentTesla",
    "Formbook":   "Formbook",
    "Loki":       "Lokibot",
    "Bladabindi": "njRAT",
    "Redline":    "Redline",
}

HIGH_SIGNAL = {
    "NtAllocateVirtualMemory", "NtProtectVirtualMemory", "NtMapViewOfSection",
    "NtUnmapViewOfSection", "NtWriteVirtualMemory", "VirtualAllocEx",
    "WriteProcessMemory", "ReadProcessMemory", "VirtualProtectEx",
    "NtReadVirtualMemory",
    "NtCreateProcess", "NtCreateProcessEx", "CreateProcess", "CreateProcessInternalW",
    "NtCreateThreadEx", "CreateRemoteThread", "CreateRemoteThreadEx",
    "NtResumeThread", "NtSuspendThread", "NtTerminateProcess", "NtTerminateThread",
    "ShellExecuteExW", "WinExec", "NtCreateSection", "NtOpenProcess",
    "RegSetValueExA", "RegSetValueExW", "RegCreateKeyExA", "RegCreateKeyExW",
    "RegOpenKeyExA", "RegOpenKeyExW", "RegDeleteKeyA", "RegDeleteKeyW",
    "RegDeleteValueA", "RegDeleteValueW", "NtSetValueKey", "NtCreateKey",
    "NtOpenKey", "NtDeleteKey",
    "InternetOpenUrl", "InternetOpenUrlA", "InternetOpenUrlW",
    "InternetConnectA", "InternetConnectW", "InternetReadFile",
    "HttpSendRequestA", "HttpSendRequestW", "HttpOpenRequestA", "HttpOpenRequestW",
    "WSASend", "WSARecv", "WSAConnect", "WSASocketA", "WSASocketW",
    "connect", "send", "recv", "socket", "bind", "listen", "accept",
    "getaddrinfo", "gethostbyname",
    "NtCreateFile", "NtOpenFile", "NtWriteFile", "NtReadFile", "NtDeleteFile",
    "CopyFileA", "CopyFileW", "MoveFileExA", "MoveFileExW",
    "DeleteFileA", "DeleteFileW", "CreateFileA", "CreateFileW",
    "NtQueryDirectoryFile", "FindFirstFileExW", "FindNextFileW",
    "CryptEncrypt", "CryptDecrypt", "CryptImportKey", "CryptExportKey",
    "CryptAcquireContextA", "CryptAcquireContextW", "CryptCreateHash",
    "CryptHashData", "CryptDeriveKey", "CryptGenKey", "CryptGetHashParam",
    "BCryptEncrypt", "BCryptDecrypt", "BCryptGenerateSymmetricKey",
    "CryptUnprotectData", "CryptProtectData",
    "CredReadA", "CredReadW", "CredEnumerateA", "CredEnumerateW",
    "LsaRetrievePrivateData", "SamOpenDatabase", "SamGetPrivateDomainInfo",
    "SetWindowsHookExA", "SetWindowsHookExW", "GetAsyncKeyState",
    "GetKeyState", "GetClipboardData", "SetClipboardData", "OpenClipboard",
    "GetForegroundWindow", "GetWindowTextA", "GetWindowTextW",
    "BitBlt", "StretchBlt", "GetDC", "CreateCompatibleBitmap",
    "PrintWindow", "capCreateCaptureWindowA",
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "NtQueryInformationProcess", "SetUnhandledExceptionFilter",
    "GetTickCount", "QueryPerformanceCounter", "NtDelayExecution",
    "OutputDebugStringA", "OutputDebugStringW",
    "GetSystemInfo", "GlobalMemoryStatusEx", "EnumProcesses",
    "GetDriveTypeA", "GetDriveTypeW", "GetLogicalDrives",
    "GetVolumeInformationA", "GetVolumeInformationW",
    "OpenProcess", "Process32FirstW", "Process32NextW",
    "CreateToolhelp32Snapshot", "Module32FirstW", "Module32NextW",
    "NtQuerySystemInformation",
    "CreateServiceA", "CreateServiceW", "OpenServiceA", "OpenServiceW",
    "StartServiceA", "StartServiceW", "ChangeServiceConfigA", "ChangeServiceConfigW",
    "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW",
    "GetProcAddress", "LdrLoadDll", "LdrGetDllHandle",
}


def extract_sequence(report_path: pathlib.Path) -> list:
    data = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
    calls = []
    for proc in data.get("behavior", {}).get("processes", []):
        for call in proc.get("calls", []):
            api = call.get("api", "")
            if api in HIGH_SIGNAL:
                calls.append((call.get("timestamp", ""), api))
    calls.sort(key=lambda x: x[0])
    return [api for _, api in calls]


def tokenize(seq: list, vocab: dict) -> list:
    tokens = [vocab.get(api, UNKNOWN_TOKEN) for api in seq]
    tokens = tokens[:SEQUENCE_LENGTH]
    tokens += [PAD_TOKEN] * (SEQUENCE_LENGTH - len(tokens))
    return tokens


def load_labels() -> dict:
    labels: dict = {}
    for label_file in [CAPE_LABELS_FILE, AVC_LABELS_FILE]:
        raw = json.loads(label_file.read_text(encoding="utf-8"))
        for dataset_label, proj_label in FAMILY_MAP.items():
            entry = raw.get(dataset_label, {})
            for r in entry.get("reports", []):
                sha = r.get("sha256", "").strip()
                if sha:
                    labels[sha] = proj_label
    return labels


def main():
    print("=" * 65)
    print("  PHENO TYPE -- Volume 4+5 Append")
    print("=" * 65)

    # Load existing vocab (frozen -- do not rebuild)
    vocab = json.loads(OUTPUT_VOCAB.read_text(encoding="utf-8"))
    print(f"\n[VOCAB] Loaded existing vocab: {len(vocab)} tokens (frozen)")

    # Load existing sha256s to avoid duplicates
    existing_shas: set = set()
    before_counts: Counter = Counter()
    with open(OUTPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_shas.add(row["sha256"])
            before_counts[row["family"]] += 1

    print(f"\n[EXISTING] {len(existing_shas)} samples already in CSV")
    for fam in sorted(before_counts):
        status = "  [CAPPED - skip]" if fam in SKIP_FAMILIES else ""
        print(f"           {fam}: {before_counts[fam]}{status}")

    # Load all labels
    labels = load_labels()
    print(f"\n[LABELS] {len(labels)} total labeled samples")

    # Index volumes 4 and 5
    new_vol_index: dict = {}
    for vol in WINMET_VOLUMES_NEW:
        if not vol.exists():
            print(f"\n[WARN] Volume not found: {vol}")
            continue
        count = 0
        for f in vol.glob("*.json"):
            new_vol_index[f.stem] = f
            count += 1
        print(f"\n[INDEX] {vol.name}: {count} report files")
    print(f"[INDEX] Combined new volumes: {len(new_vol_index)} unique files")

    # Find candidates: labeled + in new volumes + not already processed + not in skip list
    candidates: dict = {}
    for sha, fam in labels.items():
        if fam in SKIP_FAMILIES:
            continue
        if sha in existing_shas:
            continue
        if sha not in new_vol_index:
            continue
        candidates.setdefault(fam, []).append(sha)

    print(f"\n[CANDIDATES] New samples available in volumes 4+5 (excluding capped families):")
    for fam in sorted(candidates):
        print(f"             {fam}: {len(candidates[fam])}")

    if not candidates:
        print("\n[INFO] No new samples to add. Exiting.")
        return

    # Extract sequences for all candidates (no cap)
    print("\n[EXTRACT] Reading volume 3 JSON reports...")
    new_metadata = []
    new_seqs = []

    for fam in sorted(candidates):
        for sha in candidates[fam]:
            path = new_vol_index[sha]
            try:
                seq = extract_sequence(path)
                new_seqs.append(seq)
                new_metadata.append({"sha256": sha, "family": fam, "raw_len": len(seq)})
            except Exception as e:
                print(f"  [ERROR] {sha[:16]}...: {e}")

    print(f"  Extracted {len(new_seqs)} new sequences")

    if not new_seqs:
        print("\n[INFO] No sequences extracted. Exiting.")
        return

    # Append new rows to CSV (using frozen vocab)
    print(f"\n[SAVE] Appending {len(new_seqs)} rows to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for meta, seq in zip(new_metadata, new_seqs):
            w.writerow([meta["family"], meta["sha256"], meta["raw_len"]] + tokenize(seq, vocab))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    after_counts = Counter(before_counts)
    after_counts.update(m["family"] for m in new_metadata)
    new_counts = Counter(m["family"] for m in new_metadata)

    print("\n" + "=" * 65)
    print("  APPEND SUMMARY (volumes 4+5)")
    print("=" * 65)
    print(f"\n  {'Family':<15} {'Before':>7}   {'Added':>7}   {'After':>7}")
    print(f"  {'-'*15} {'-'*7}   {'-'*7}   {'-'*7}")
    for fam in sorted(set(list(before_counts.keys()) + list(new_counts.keys()))):
        b = before_counts.get(fam, 0)
        a = new_counts.get(fam, 0)
        print(f"  {fam:<15} {b:>7}   {a:>7}   {b+a:>7}")

    total_before = sum(before_counts.values())
    total_added  = len(new_metadata)
    total_after  = total_before + total_added

    print(f"\n  Total rows before   : {total_before}")
    print(f"  New rows added      : {total_added}")
    print(f"  Total rows after    : {total_after}")

    under_200 = [fam for fam, cnt in after_counts.items() if cnt < 200]
    if under_200:
        print(f"\n  [WARN] Families still below 200 samples:")
        for fam in sorted(under_200):
            print(f"         {fam}: {after_counts[fam]}")
    else:
        print("\n  All families are at or above 200 samples.")

    print(f"\n  Output CSV  : {OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
