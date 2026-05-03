"""
PHENO TYPE -- Held-Out Family Extraction
==========================================
Extracts 1200-token sequences for held-out (test) families not seen during training.

Held-out families:
  Amadey, Dacic, Smokeloader, Remcos, Qakbot

Rules:
  - Frozen vocab from final_dna_v2_vocab.json (no rebuild ever)
  - Same HIGH_SIGNAL filter as training pipeline
  - Up to MAX_PER_FAMILY per family (all if fewer available)
  - Never include sha256 already in final_dna_v2.csv
  - Appends to held_out_families.csv if it exists (for multi-volume runs)
  - Scans whichever WinMET volumes exist on disk
  - Label source: cape + avclass (union)
"""

import json, csv, pathlib, sys
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH  = 1200
PAD_TOKEN        = 0
UNKNOWN_TOKEN    = 1
MAX_PER_FAMILY   = 50

BASE = pathlib.Path(__file__).parent

# Scan all volumes that exist
ALL_VOLUMES = [
    BASE / "WinMET_volume_1",
    BASE / "WinMET_volume_2",
    BASE / "WinMET_volume_3",
    BASE / "WinMET_volume_4",
    BASE / "WinMET_volume_5",
]

LABEL_DIR        = BASE / "Winmet  DataSet Stuff"
CAPE_LABELS_FILE = LABEL_DIR / "cape_report_to_label_mapping.json"
AVC_LABELS_FILE  = LABEL_DIR / "avclass_report_to_label_mapping.json"

TRAINING_CSV  = BASE / "final_dna_v2.csv"
OUTPUT_CSV    = BASE / "held_out_families.csv"
VOCAB_FILE    = BASE / "final_dna_v2_vocab.json"

# Dataset label -> project label for held-out families
HELD_OUT_MAP = {
    "Amadey":      "Amadey",
    "Dacic":       "Dacic",
    "Smokeloader": "Smokeloader",
    "Remcos":      "Remcos",
    "Qakbot":      "Qakbot",
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

HEADERS = ["family", "sha256", "raw_api_len"] + [f"tok_{i}" for i in range(SEQUENCE_LENGTH)]


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


def load_held_out_labels() -> dict:
    """Returns {sha256: family_name} for all held-out families, union of both label files."""
    labels: dict = {}
    for label_file in [CAPE_LABELS_FILE, AVC_LABELS_FILE]:
        raw = json.loads(label_file.read_text(encoding="utf-8"))
        for dataset_label, proj_label in HELD_OUT_MAP.items():
            for r in raw.get(dataset_label, {}).get("reports", []):
                sha = r.get("sha256", "").strip()
                if sha:
                    labels[sha] = proj_label
    return labels


def build_volume_index() -> dict:
    """Scan all available WinMET volumes. Returns {sha256: path}."""
    index = {}
    for vol in ALL_VOLUMES:
        if not vol.exists():
            continue
        count = 0
        for f in vol.glob("*.json"):
            index[f.stem] = f
            count += 1
        print(f"  [VOL] {vol.name}: {count} files")
    return index


def main():
    print("=" * 65)
    print("  PHENO TYPE -- Held-Out Family Extraction")
    print("=" * 65)

    # Load frozen vocab
    vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    print(f"\n[VOCAB] Frozen vocab loaded: {len(vocab)} tokens")

    # Load sha256s to exclude (training set + already extracted held-out)
    excluded_shas: set = set()
    with open(TRAINING_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            excluded_shas.add(row["sha256"])
    print(f"[EXCLUDE] {len(excluded_shas)} training samples will be skipped")

    # Also exclude any already in held_out_families.csv
    before_counts: Counter = Counter()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                excluded_shas.add(row["sha256"])
                before_counts[row["family"]] += 1
        print(f"[EXCLUDE] +{sum(before_counts.values())} already in held_out_families.csv")
        print(f"          Running totals before this run:")
        for fam in sorted(before_counts):
            print(f"            {fam}: {before_counts[fam]}")
    else:
        print("[OUTPUT] held_out_families.csv does not exist yet -- will create")

    # Load labels for held-out families
    labels = load_held_out_labels()
    print(f"\n[LABELS] {len(labels)} held-out samples labeled across all families")
    fam_label_counts = Counter(labels.values())
    for fam in sorted(fam_label_counts):
        print(f"         {fam}: {fam_label_counts[fam]} labeled")

    # Build volume index (only existing volumes)
    print("\n[INDEX] Scanning available WinMET volumes...")
    vol_index = build_volume_index()
    print(f"        Total indexed: {len(vol_index)} files")

    # Find candidates: labeled + on disk + not excluded + not yet at cap
    candidates: dict = {}
    for sha, fam in labels.items():
        if sha in excluded_shas:
            continue
        if sha not in vol_index:
            continue
        already_have = before_counts.get(fam, 0)
        if already_have >= MAX_PER_FAMILY:
            continue
        candidates.setdefault(fam, []).append(sha)

    # Trim to remaining cap space
    for fam in list(candidates.keys()):
        already_have = before_counts.get(fam, 0)
        remaining    = MAX_PER_FAMILY - already_have
        if len(candidates[fam]) > remaining:
            candidates[fam] = candidates[fam][:remaining]

    print(f"\n[CANDIDATES] New samples available this run:")
    for fam in sorted(HELD_OUT_MAP.values()):
        n = len(candidates.get(fam, []))
        already = before_counts.get(fam, 0)
        print(f"             {fam}: {n} new  (already have: {already})")

    total_candidates = sum(len(v) for v in candidates.values())
    if total_candidates == 0:
        print("\n[INFO] No new samples to extract. All families at cap or no files on disk.")
        return

    # Extract sequences
    print(f"\n[EXTRACT] Reading {total_candidates} JSON reports...")
    new_metadata = []
    new_seqs     = []

    for fam in sorted(candidates):
        for sha in candidates[fam]:
            path = vol_index[sha]
            try:
                seq = extract_sequence(path)
                new_seqs.append(seq)
                new_metadata.append({"sha256": sha, "family": fam, "raw_len": len(seq)})
            except Exception as e:
                print(f"  [ERROR] {sha[:16]}...: {e}")

    print(f"  Extracted {len(new_seqs)} sequences")

    # Write / append to output CSV
    file_exists = OUTPUT_CSV.exists()
    mode = "a" if file_exists else "w"
    action = "Appending to" if file_exists else "Creating"
    print(f"\n[SAVE] {action} {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADERS)
        for meta, seq in zip(new_metadata, new_seqs):
            w.writerow([meta["family"], meta["sha256"], meta["raw_len"]] + tokenize(seq, vocab))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    after_counts  = Counter(before_counts)
    after_counts.update(m["family"] for m in new_metadata)
    new_counts    = Counter(m["family"] for m in new_metadata)

    print("\n" + "=" * 65)
    print("  HELD-OUT EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"\n  {'Family':<15} {'Before':>7}   {'Added':>7}   {'After':>7}   {'Avg active':>10}")
    print(f"  {'-'*15} {'-'*7}   {'-'*7}   {'-'*7}   {'-'*10}")

    for fam in sorted(HELD_OUT_MAP.values()):
        b   = before_counts.get(fam, 0)
        add = new_counts.get(fam, 0)
        aft = after_counts.get(fam, 0)
        fam_new = [m for m in new_metadata if m["family"] == fam]
        if fam_new:
            avg = sum(m["raw_len"] for m in fam_new) / len(fam_new)
            avg_str = f"{avg:>10.1f}"
        else:
            avg_str = "       n/a"
        print(f"  {fam:<15} {b:>7}   {add:>7}   {aft:>7}   {avg_str}")

    total_added = len(new_metadata)
    total_after = sum(after_counts.values())
    print(f"\n  New rows added this run : {total_added}")
    print(f"  Total held-out rows     : {total_after}")
    print(f"  Cap per family          : {MAX_PER_FAMILY}")

    under_cap = [(fam, after_counts.get(fam, 0)) for fam in sorted(HELD_OUT_MAP.values())
                 if after_counts.get(fam, 0) < MAX_PER_FAMILY]
    if under_cap:
        print(f"\n  Families still below cap ({MAX_PER_FAMILY}):")
        for fam, cnt in under_cap:
            print(f"    {fam}: {cnt}/{MAX_PER_FAMILY}")
    else:
        print(f"\n  All families at or above cap ({MAX_PER_FAMILY}).")

    print(f"\n  Output : {OUTPUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
