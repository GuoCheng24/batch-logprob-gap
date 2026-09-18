"""Every `TABLES.md#anchor` the README links must match a heading in TABLES.md.

GitHub builds a heading's anchor by lower-casing it, dropping characters that are neither word
characters, spaces nor hyphens, and turning each remaining space into a hyphen -- so "## T1  dtype x
batch" (two spaces) becomes `t1--dtype-x-batch`. Rebuilding TABLES.md with a reworded heading changes
the anchor and leaves the README pointing at nothing, which a reader only notices by landing at the
top of the file. Verified against the ids GitHub actually emits for this file.

    python scripts/check_table_anchors.py
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def anchor(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")


def main() -> int:
    tables = (ROOT / "TABLES.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    have = {anchor(h) for h in re.findall(r"^## (.+)$", tables, re.M)}
    used = re.findall(r"TABLES\.md#([\w-]+)", readme)
    missing = sorted({u for u in used if u not in have})
    print(f"  {len(used)} links into TABLES.md, {len(have)} headings")
    for m in missing:
        print(f"  no heading produces #{m}")
    if missing:
        return 1
    print("  every table link resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
