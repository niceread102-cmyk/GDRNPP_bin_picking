"""Git clean filter: notebook JSON on stdin -> the same notebook without outputs on stdout.

The working copy keeps its outputs (figures, tables); only what git stores is stripped, so the repository stays
small (results_photoneo_summary.ipynb: 45 MB with outputs). Enabled by .gitattributes (*.ipynb filter=strip-notebook-output)
once the filter is defined in the local git config (setup_venv.sh / setup_venv.ps1 do this):

    git config filter.strip-notebook-output.clean "python3 tools/strip_notebook_outputs.py"

Files can also be stripped in place:  python3 tools/strip_notebook_outputs.py notebooks/*.ipynb
"""
from __future__ import annotations

import json
import sys


def strip(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def dumps(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main():
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            with open(path, encoding="utf-8") as f:
                nb = json.load(f)
            with open(path, "w", encoding="utf-8") as f:
                f.write(dumps(strip(nb)))
            print(f"stripped {path}", file=sys.stderr)
        return
    data = sys.stdin.buffer.read().decode("utf-8")
    try:
        nb = json.loads(data)
    except json.JSONDecodeError:            # not a notebook (or mid-merge): pass it through unchanged
        sys.stdout.buffer.write(data.encode("utf-8"))
        return
    sys.stdout.buffer.write(dumps(strip(nb)).encode("utf-8"))


if __name__ == "__main__":
    main()
