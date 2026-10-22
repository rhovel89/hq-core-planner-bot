#!/usr/bin/env python3
"""Fail-closed, read-only startup check for the original Discord HQ planner.

Does not contact Discord, alter JSON, or print tokens, guild IDs or player names.
Used before the Railway worker starts so a missing volume cannot silently
initialize an empty in-memory map and accidentally overwrite historic layouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def check(require_volume: bool = False, inspect_only: bool = False) -> int:
    path = Path(os.environ.get("DATA_FILE", "hq_layouts.json")).expanduser()
    resolved = path.resolve()
    mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    print("HQ preflight: INSPECT ONLY" if inspect_only else "HQ preflight: READ ONLY")
    print(f"DATA_FILE={resolved}")

    if not inspect_only and not os.environ.get("DISCORD_BOT_TOKEN", "").strip():
        print("FAIL: DISCORD_BOT_TOKEN is not configured (value not displayed)")
        return 2
    if require_volume:
        if not mount:
            print("FAIL: no Railway volume mount path is reported")
            return 3
        root = Path(mount).resolve()
        if root not in resolved.parents:
            print("FAIL: DATA_FILE is not inside the mounted volume")
            return 3
    if not resolved.is_file():
        print("FAIL: HQ data file does not exist; original layouts were not altered")
        return 4
    try:
        raw = resolved.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: HQ file unreadable or invalid JSON ({type(exc).__name__})")
        return 5
    if not isinstance(data, dict) or not data:
        print("FAIL: file has no guild-keyed layout records")
        return 6
    for record in data.values():
        if not isinstance(record, dict) or not isinstance(record.get("claims"), dict):
            print("FAIL: at least one guild layout is malformed")
            return 6
    claim_count = sum(len(record["claims"]) for record in data.values())
    print(f"PASS: {len(data)} guild layout(s), {claim_count} saved assignment(s)")
    print(f"File bytes: {len(raw)}; SHA-256: {hashlib.sha256(raw).hexdigest()}")
    print("Only schema shape was checked. Live coordinates and guild mapping still need verification.")
    if inspect_only:
        print("INSPECTION COMPLETE: no Discord connection, no bot token required, no files modified.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-volume", action="store_true",
                        help="Fail unless the JSON path is inside the Railway volume mount")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Read and fingerprint existing layout without needing Discord credentials or starting bot")
    args = parser.parse_args()
    raise SystemExit(check(require_volume=args.require_volume, inspect_only=args.inspect_only))
