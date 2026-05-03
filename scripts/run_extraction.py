"""
PHENO TYPE -- Data Extraction Pipeline v2 (WinMET-only, 5 families)
====================================================================
Families:
  AgentTesla  (CAPE/avclass label: "Agenttesla")
  Formbook    (CAPE/avclass label: "Formbook")
  Lokibot     (CAPE/avclass label: "Loki")
  njRAT       (CAPE/avclass label: "Bladabindi")
  Redline     (CAPE/avclass label: "Redline")

Label sources (union of both -- more coverage):
  cape_report_to_label_mapping.json
  avclass_report_to_label_mapping.json

Report files: WinMET_volume_1/ + WinMET_volume_2/

Output: final_dna_v2.csv  (family, sha256, raw_api_len, tok_0 .. tok_1199)
        final_dna_v2_vocab.json
"""

import json, csv, random, pathlib, sys
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH    = 1200
PAD_TOKEN          = 0
UNKNOWN_TOKEN      = 1
MAX_PER_FAMILY     = 500   # cap per family; takes all available if fewer
RANDOM_SEED        = 42

BASE = pathlib.Path(__file__).parent

WINMET_VOLUMES    = [BASE / "WinMET_volume_1", BASE / "WinMET_volume_2"]
LABEL_DIR         = BASE / "Winmet  DataSet Stuff"
CAPE_LABELS_FILE  = LABEL_DIR / "cape_report_to_label_mapping.json"
AVC_LABELS_FILE   = LABEL_DIR / "avclass_report_to_label_mapping.json"

OUTPUT_CSV        = BASE / "final_dna_v2.csv"
OUTPUT_VOCAB      = BASE / "final_dna_v2_vocab.json"

# Dataset label -> project label
FAMILY_MAP = {
    "Agenttesla": "AgentTesla",
    "Formbook":   "Formbook",
    "Loki":       "Lokibot",
    "Bladabindi": "njRAT",
    "Redline":    "Redline",
}

# HIGH_SIGNAL: only these API calls are kept from the dynamic traces.
HIGH_SIGNAL = {
    # === MEMORY INJECTION (all families use this for unpacking/hollowing) ===
    "NtAllocateVirtualMemory", "NtProtectVirtualMemory", "NtMapViewOfSection",
    "NtUnmapViewOfSection", "NtWriteVirtualMemory", "VirtualAllocEx",
    "WriteProcessMemory", "ReadProcessMemory", "VirtualProtectEx",
    "NtReadVirtualMemory",

    # === PROCESS & THREAD CONTROL (process hollowing, injection, spawning) ===
    "NtCreateProcess", "NtCreateProcessEx", "CreateProcess", "CreateProcessInternalW",
    "NtCreateThreadEx", "CreateRemoteThread", "CreateRemoteThreadEx",
    "NtResumeThread", "NtSuspendThread", "NtTerminateProcess", "NtTerminateThread",
    "ShellExecuteExW", "WinExec", "NtCreateSection", "NtOpenProcess",

    # === REGISTRY -- PERSISTENCE (all RATs and stealers write run keys) ===
    "RegSetValueExA", "RegSetValueExW", "RegCreateKeyExA", "RegCreateKeyExW",
    "RegOpenKeyExA", "RegOpenKeyExW", "RegDeleteKeyA", "RegDeleteKeyW",
    "RegDeleteValueA", "RegDeleteValueW", "NtSetValueKey", "NtCreateKey",
    "NtOpenKey", "NtDeleteKey",

    # === NETWORK -- C2 COMMUNICATION (Lokibot HTTP, AgentTesla SMTP, njRAT TCP) ===
    "InternetOpenUrl", "InternetOpenUrlA", "InternetOpenUrlW",
    "InternetConnectA", "InternetConnectW", "InternetReadFile",
    "HttpSendRequestA", "HttpSendRequestW", "HttpOpenRequestA", "HttpOpenRequestW",
    "WSASend", "WSARecv", "WSAConnect", "WSASocketA", "WSASocketW",
    "connect", "send", "recv", "socket", "bind", "listen", "accept",
    "getaddrinfo", "gethostbyname",

    # === FILE SYSTEM -- DROPPER & PAYLOAD (dropping files, deleting evidence) ===
    "NtCreateFile", "NtOpenFile", "NtWriteFile", "NtReadFile", "NtDeleteFile",
    "CopyFileA", "CopyFileW", "MoveFileExA", "MoveFileExW",
    "DeleteFileA", "DeleteFileW", "CreateFileA", "CreateFileW",
    "NtQueryDirectoryFile", "FindFirstFileExW", "FindNextFileW",

    # === CRYPTOGRAPHY (payload decryption, credential encryption) ===
    "CryptEncrypt", "CryptDecrypt", "CryptImportKey", "CryptExportKey",
    "CryptAcquireContextA", "CryptAcquireContextW", "CryptCreateHash",
    "CryptHashData", "CryptDeriveKey", "CryptGenKey", "CryptGetHashParam",
    "BCryptEncrypt", "BCryptDecrypt", "BCryptGenerateSymmetricKey",

    # === CREDENTIAL THEFT (AgentTesla, Redline, Lokibot specific) ===
    "CryptUnprotectData", "CryptProtectData",
    "CredReadA", "CredReadW", "CredEnumerateA", "CredEnumerateW",
    "LsaRetrievePrivateData", "SamOpenDatabase", "SamGetPrivateDomainInfo",

    # === KEYLOGGING & INPUT CAPTURE (AgentTesla, njRAT) ===
    "SetWindowsHookExA", "SetWindowsHookExW", "GetAsyncKeyState",
    "GetKeyState", "GetClipboardData", "SetClipboardData", "OpenClipboard",
    "GetForegroundWindow", "GetWindowTextA", "GetWindowTextW",

    # === SCREENSHOT CAPTURE (njRAT webcam/screen, AgentTesla screenshots) ===
    "BitBlt", "StretchBlt", "GetDC", "CreateCompatibleBitmap",
    "PrintWindow", "capCreateCaptureWindowA",

    # === ANTI-ANALYSIS & EVASION (all families check for sandboxes) ===
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "NtQueryInformationProcess", "SetUnhandledExceptionFilter",
    "GetTickCount", "QueryPerformanceCounter", "NtDelayExecution",
    "OutputDebugStringA", "OutputDebugStringW",
    "GetSystemInfo", "GlobalMemoryStatusEx", "EnumProcesses",

    # === USB / DRIVE SPREADING (njRAT specific -- spreads via USB) ===
    "GetDriveTypeA", "GetDriveTypeW", "GetLogicalDrives",
    "GetVolumeInformationA", "GetVolumeInformationW",

    # === PROCESS ENUMERATION (Redline scans for browsers, crypto wallets) ===
    "OpenProcess", "Process32FirstW", "Process32NextW",
    "CreateToolhelp32Snapshot", "Module32FirstW", "Module32NextW",
    "NtQuerySystemInformation",

    # === SERVICE & STARTUP MANIPULATION (persistence mechanisms) ===
    "CreateServiceA", "CreateServiceW", "OpenServiceA", "OpenServiceW",
    "StartServiceA", "StartServiceW", "ChangeServiceConfigA", "ChangeServiceConfigW",

    # === LIBRARY LOADING (unpacking, dynamic API resolution) ===
    "LoadLibraryA", "LoadLibraryW", "LoadLibraryExA", "LoadLibraryExW",
    "GetProcAddress", "LdrLoadDll", "LdrGetDllHandle",
}


# ---------------------------------------------------------------------------
# 1. Sequence extraction
# ---------------------------------------------------------------------------
def extract_sequence(report_path: pathlib.Path) -> list:
    """
    Read a WinMET/CAPE JSON report.
    Return time-ordered list of HIGH_SIGNAL API call names across all processes.
    Timestamps are 'YYYY-MM-DD HH:MM:SS,mmm' strings -- lexicographic sort is correct.
    """
    data = json.loads(report_path.read_text(encoding="utf-8", errors="replace"))
    calls = []
    for proc in data.get("behavior", {}).get("processes", []):
        for call in proc.get("calls", []):
            api = call.get("api", "")
            if api in HIGH_SIGNAL:
                calls.append((call.get("timestamp", ""), api))
    calls.sort(key=lambda x: x[0])
    return [api for _, api in calls]


# ---------------------------------------------------------------------------
# 2. Label loading (union of CAPE + avclass mappings)
# ---------------------------------------------------------------------------
def load_labels() -> dict:
    """
    Returns {sha256: project_family} combining both label files.
    Both files share the same nested structure:
      { DatasetLabel: { n_reports: N, reports: [{sha256: hash, ...}, ...] } }
    """
    labels: dict = {}

    for label_file in [CAPE_LABELS_FILE, AVC_LABELS_FILE]:
        raw = json.loads(label_file.read_text(encoding="utf-8"))
        for dataset_label, proj_label in FAMILY_MAP.items():
            entry = raw.get(dataset_label, {})
            for r in entry.get("reports", []):
                sha = r.get("sha256", "").strip()
                if sha:
                    labels[sha] = proj_label   # last write wins (both agree)

    print(f"\n[LABELS] Combined labels loaded: {len(labels)} unique samples")
    for proj in sorted(set(FAMILY_MAP.values())):
        n = sum(1 for v in labels.values() if v == proj)
        print(f"         {proj}: {n} labeled")
    return labels


# ---------------------------------------------------------------------------
# 3. Volume index
# ---------------------------------------------------------------------------
def build_index() -> dict:
    """Scan both WinMET volumes. Returns {sha256: pathlib.Path}."""
    index = {}
    for vol in WINMET_VOLUMES:
        if not vol.exists():
            print(f"[WARN] Volume not found: {vol}")
            continue
        for f in vol.glob("*.json"):
            index[f.stem] = f
    print(f"\n[INDEX] {len(index)} report files indexed across {len(WINMET_VOLUMES)} volumes")
    return index


# ---------------------------------------------------------------------------
# 4. Vocabulary
# ---------------------------------------------------------------------------
def build_vocab(all_sequences: list) -> dict:
    freq = Counter(api for seq in all_sequences for api in seq)
    print(f"\n[VOCAB] {len(freq)} unique API tokens found")
    print("[VOCAB] Top 15 by frequency:")
    for api, cnt in freq.most_common(15):
        flag = "  *" if api in HIGH_SIGNAL else ""
        print(f"        {api:<50} {cnt:>7}{flag}")
    vocab = {"<PAD>": PAD_TOKEN, "<UNK>": UNKNOWN_TOKEN}
    for i, (api, _) in enumerate(freq.most_common(), start=2):
        vocab[api] = i
    return vocab


def tokenize(seq: list, vocab: dict) -> list:
    tokens = [vocab.get(api, UNKNOWN_TOKEN) for api in seq]
    tokens = tokens[:SEQUENCE_LENGTH]
    tokens += [PAD_TOKEN] * (SEQUENCE_LENGTH - len(tokens))
    return tokens


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(RANDOM_SEED)

    print("=" * 65)
    print("  PHENO TYPE -- Extraction Pipeline")
    print("=" * 65)

    labels = load_labels()
    index  = build_index()

    # Group available (labeled + on disk) by family
    by_family: dict = {}
    for sha, fam in labels.items():
        if sha in index:
            by_family.setdefault(fam, []).append(sha)

    print("\n[CHECK] Files on disk per family:")
    for fam in sorted(by_family):
        print(f"        {fam}: {len(by_family[fam])}")

    # Select up to MAX_PER_FAMILY per family
    selected: dict = {}
    for fam, shas in by_family.items():
        n = min(len(shas), MAX_PER_FAMILY)
        selected[fam] = rng.sample(shas, n)
        if len(shas) < 20:
            print(f"[WARN]  {fam}: only {len(shas)} found -- under 20 minimum")

    # Extract sequences
    print("\n[EXTRACT] Reading JSON reports...")
    raw_seqs = []
    metadata = []

    for fam in sorted(selected):
        for sha in selected[fam]:
            path = index[sha]
            try:
                seq = extract_sequence(path)
                raw_seqs.append(seq)
                metadata.append({"sha256": sha, "family": fam, "raw_len": len(seq)})
            except Exception as e:
                print(f"  [ERROR] {sha[:16]}...: {e}")

    if not raw_seqs:
        print("\n[ERROR] No sequences extracted. Check paths and labels.")
        sys.exit(1)

    print(f"  Extracted {len(raw_seqs)} sequences total")

    # Build shared vocabulary
    vocab = build_vocab(raw_seqs)

    # Write CSV
    print(f"\n[SAVE] Writing {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        headers = ["family", "sha256", "raw_api_len"] + [f"tok_{i}" for i in range(SEQUENCE_LENGTH)]
        w = csv.writer(f)
        w.writerow(headers)
        for meta, seq in zip(metadata, raw_seqs):
            w.writerow([meta["family"], meta["sha256"], meta["raw_len"]] + tokenize(seq, vocab))

    OUTPUT_VOCAB.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY v2")
    print("=" * 65)

    fam_counts = Counter(m["family"] for m in metadata)
    print(f"\n  {'Family':<15} {'Rows':>5}   {'Avg active tokens':>18}   {'Cap hits (>=1200)':>18}")
    print(f"  {'-'*15} {'-'*5}   {'-'*18}   {'-'*18}")
    for fam in sorted(fam_counts):
        fam_meta = [m for m in metadata if m["family"] == fam]
        avg_nz   = sum(m["raw_len"] for m in fam_meta) / len(fam_meta)
        cap_hits = sum(1 for m in fam_meta if m["raw_len"] >= SEQUENCE_LENGTH)
        print(f"  {fam:<15} {fam_counts[fam]:>5}   {avg_nz:>18.1f}   {cap_hits:>18}")

    total   = len(metadata)
    overall = sum(m["raw_len"] for m in metadata) / total
    total_cap = sum(1 for m in metadata if m["raw_len"] >= SEQUENCE_LENGTH)
    print(f"\n  Total rows          : {total}")
    print(f"  Vocabulary size     : {len(vocab)}  (0=PAD, 1=UNK, 2+= API names)")
    print(f"  Sequence length     : {SEQUENCE_LENGTH}")
    print(f"  Avg active/sample   : {overall:.1f}")
    print(f"  Total cap hits      : {total_cap} samples hit >= {SEQUENCE_LENGTH} token cap")

    # Which HIGH_SIGNAL tokens were actually seen in the data
    seen_tokens = set(vocab.keys()) - {"<PAD>", "<UNK>"}
    not_seen    = HIGH_SIGNAL - seen_tokens
    print(f"\n  HIGH_SIGNAL tokens in filter  : {len(HIGH_SIGNAL)}")
    print(f"  Tokens actually found in data : {len(seen_tokens)}")
    print(f"  Tokens NOT found in data      : {len(not_seen)}")
    if not_seen:
        print("    (not matched in any report):")
        for api in sorted(not_seen):
            print(f"      {api}")

    print(f"\n  Output CSV  : {OUTPUT_CSV}")
    print(f"  Vocab JSON  : {OUTPUT_VOCAB}")
    print("=" * 65)

    # Sanity warnings
    missing = [f for f in FAMILY_MAP.values() if f not in fam_counts]
    if missing:
        print(f"\n[WARN] Missing families in output: {missing}")
    under = [f for f, c in fam_counts.items() if c < 20]
    if under:
        print(f"[WARN] Families under 20 samples: {under}")
    low_signal = [m for m in metadata if m["raw_len"] < 5]
    if low_signal:
        print(f"[WARN] {len(low_signal)} samples have fewer than 5 high-signal calls")
        print("       Consider broadening HIGH_SIGNAL if this is most of a family.")


if __name__ == "__main__":
    main()
