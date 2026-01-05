# generate_theorems_md_from_ipynb.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import nbformat

# Optional YAML parser for _toc.yml ordering; fallback to regex parsing if unavailable.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ----------------------------
# What we extract
# ----------------------------
PRF_TYPES = ("theorem", "lemma", "corollary")  # add more if you like


@dataclass(frozen=True)
class PrfItem:
    typ: str  # theorem / lemma / corollary
    label: str
    title: str
    body: str
    source: str  # notebook path (relative)


# Matches fenced directives like:
# ```{prf:theorem} Title
# :label: thm-foo
# ...
# ```
# Works with both ``` and ~~~ fences.
PRF_BLOCK_RE = re.compile(
    rf"""
(?P<fence>```|~~~)                         # opening fence
\{{prf:(?P<typ>{'|'.join(PRF_TYPES)})\}}   # directive type
[ \t]*(?P<title>[^\n\r]*)                  # title on same line (optional)
[\r\n]+
(?P<body>.*?)
^(?P=fence)[ \t]*$                         # closing fence
""",
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)

LABEL_RE = re.compile(r"^\s*:label:\s*(?P<label>\S+)\s*$", re.MULTILINE)

TOC_FILE_LINE_RE = re.compile(r"^\s*file\s*:\s*(?P<file>[^\s#]+)\s*$", re.MULTILINE)


def extract_prf_items_from_markdown(md: str) -> List[tuple[str, str, str, str]]:
    """
    Returns list of (typ, label, title, body_content_only).
    body_content_only excludes the ':label:' line.
    """
    out: List[tuple[str, str, str, str]] = []
    for m in PRF_BLOCK_RE.finditer(md):
        typ = m.group("typ").strip().lower()
        title = (m.group("title") or "").strip()
        full_body = m.group("body")

        lm = LABEL_RE.search(full_body)
        if not lm:
            continue
        label = lm.group("label").strip()

        # Remove the first :label: line
        body_wo_label = LABEL_RE.sub("", full_body, count=1).strip()

        if not title:
            title = label

        out.append((typ, label, title, body_wo_label))
    return out


def extract_from_notebook(nb_path: Path, root: Path) -> List[PrfItem]:
    nb = nbformat.read(nb_path, as_version=4)
    rel = nb_path.relative_to(root).as_posix()

    items: List[PrfItem] = []
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        md = cell.get("source") or ""
        for typ, label, title, body in extract_prf_items_from_markdown(md):
            items.append(PrfItem(typ=typ, label=label, title=title, body=body, source=rel))
    return items


def _flatten_toc_yaml(obj) -> List[str]:
    """Pull 'file' values from Jupyter Book toc YAML structure in-order."""
    out: List[str] = []
    if isinstance(obj, dict):
        if "file" in obj and isinstance(obj["file"], str):
            out.append(obj["file"])
        for k in ("chapters", "sections", "parts", "entries", "subtrees"):
            if k in obj:
                out.extend(_flatten_toc_yaml(obj[k]))
        for v in obj.values():
            out.extend(_flatten_toc_yaml(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_flatten_toc_yaml(it))
    return out


def read_toc_files_in_order(toc_path: Path) -> List[str]:
    """
    Returns a list of 'file' paths from _toc.yml in the order Jupyter Book uses.
    Paths are as written in _toc.yml (usually without extension).
    """
    text = toc_path.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text)
        files = _flatten_toc_yaml(data)
        seen = set()
        out: List[str] = []
        for f in files:
            if f not in seen:
                out.append(f)
                seen.add(f)
        return out

    files = [m.group("file") for m in TOC_FILE_LINE_RE.finditer(text)]
    seen = set()
    out: List[str] = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def resolve_toc_file_to_ipynb(root: Path, toc_file: str) -> Optional[Path]:
    """
    Map a toc entry like 'notebooks/InfDis/InfDisSizeDist' to an existing .ipynb.
    """
    p = root / toc_file
    if p.suffix:
        return p if (p.exists() and p.suffix == ".ipynb") else None

    ipynb = p.with_suffix(".ipynb")
    return ipynb if ipynb.exists() else None


def write_theorems_md(out_path: Path, items: List[PrfItem]) -> None:
    lines: List[str] = []
    lines.append("# List of Theorems")
    lines.append("")
    lines.append(
        "*(Auto-generated from source notebooks in `_toc.yml` order. "
        "To update: run `python generate_theorems_md_from_ipynb.py` and rebuild.)*"
    )
    lines.append("")

    # Interleaved in book order: theorem/lemma/corollary appear exactly where they occur.
    for it in items:
        # {prf:ref} renders the correct type+number (Theorem/Lemma/Corollary ...)
        lines.append(f"## {{prf:ref}}`{it.label}` — {it.title}")
        lines.append("")
#        lines.append(f"*Source:* `{it.source}`")
#        lines.append("")
        lines.append(":::{admonition} Statement")
        lines.append(":class: dropdown")
        lines.append("")
        lines.append(it.body.strip())
        lines.append(":::")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(".").resolve()
    toc_path = root / "_toc.yml"
    if not toc_path.exists():
        raise SystemExit("Could not find _toc.yml at book root.")

    toc_files = read_toc_files_in_order(toc_path)

    all_items: List[PrfItem] = []
    missing = []
    for f in toc_files:
        nb_path = resolve_toc_file_to_ipynb(root, f)
        if nb_path is None:
            continue
        if not nb_path.exists():
            missing.append(f)
            continue
        all_items.extend(extract_from_notebook(nb_path, root))

    out_path = root / "theorems.md"
    write_theorems_md(out_path, all_items)

    print(f"Wrote {out_path} with {len(all_items)} items ({', '.join(PRF_TYPES)}).")
    if missing:
        print("Warning: these TOC entries did not resolve to an .ipynb:")
        for f in missing:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
