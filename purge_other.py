"""
Purge "Other"-classified entries from the analysis cache so they are
re-processed on the next analysis run.

Targets:
  - Entries where issue_category == "Other" (includes parse-error fallbacks)

All other cached entries are left untouched.

Usage:
    python purge_other.py            # dry run — shows what would be removed
    python purge_other.py --confirm  # writes the updated cache to disk
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "cache" / "analysis_cache.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge Other entries from analysis cache.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually write the purged cache. Without this flag the script is a dry run.",
    )
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        print(f"Cache not found at {CACHE_PATH}. Nothing to do.")
        return

    with open(CACHE_PATH) as fh:
        cache: dict = json.load(fh)

    to_purge = [
        rid for rid, v in cache.items()
        if v.get("issue_category") == "Other"
    ]
    parse_errors = [rid for rid in to_purge if cache[rid].get("_parse_error")]

    print(f"Cache total        : {len(cache)} entries")
    print(f"To purge (Other)   : {len(to_purge)} entries")
    print(f"  of which errors  : {len(parse_errors)} parse-error fallbacks")
    print(f"Remaining after    : {len(cache) - len(to_purge)} entries")
    print()

    if not to_purge:
        print("Nothing to purge.")
        return

    if not args.confirm:
        print("DRY RUN — no changes written. Re-run with --confirm to apply.")
        print()
        print("Entries that would be removed:")
        for rid in to_purge:
            v = cache[rid]
            flag = " [parse error]" if v.get("_parse_error") else ""
            print(f"  {rid}{flag}")
        return

    # Back up the cache before modifying
    backup_path = CACHE_PATH.with_suffix(
        f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy(CACHE_PATH, backup_path)
    print(f"Backup written to: {backup_path}")

    # Remove the Other entries
    for rid in to_purge:
        del cache[rid]

    with open(CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)

    print(f"Purged {len(to_purge)} entries. Cache now has {len(cache)} entries.")
    print("Re-run analysis in the app to reclassify the purged responses.")


if __name__ == "__main__":
    main()
