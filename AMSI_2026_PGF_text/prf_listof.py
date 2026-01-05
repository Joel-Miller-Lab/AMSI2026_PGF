# prf_listof.py
from __future__ import annotations

import re
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.addnodes import pending_xref


class prf_listof_node(nodes.General, nodes.Element):
    """Placeholder node that we fill at doctree-resolved."""


class PrfListOf(Directive):
    """
    MyST usage:

    ~~~{prf:listof} theorem
    :sort: number   # default
    :debug:
    ~~~

    sort options:
      :sort: number   -> by resolved theorem number (e.g. 2.4, 3.2, 10.1)
      :sort: title    -> alphabetical by title
      :sort: none     -> keep document order
    """

    required_arguments = 1
    has_content = False
    option_spec = {
        "sort": directives.unchanged,
        "debug": directives.flag,
    }

    def run(self):
        want_type = self.arguments[0].strip().lower()
        node = prf_listof_node()
        node["want_type"] = want_type
        node["sort"] = (self.options.get("sort") or "number").strip().lower()
        node["debug"] = "debug" in self.options
        return [node]


def _node_looks_like(node: nodes.Node, want_type: str) -> bool:
    classes = [c.lower() for c in node.get("classes", [])]
    if not classes:
        return False
    if want_type in classes:
        return True
    if f"prf-{want_type}" in classes:
        return True
    if "admonition" in classes and want_type in classes:
        return True
    if "prf" in classes and want_type in classes:
        return True
    return False


def _find_anchor_id(node: nodes.Node) -> str | None:
    ids = node.get("ids", [])
    if ids:
        return ids[0]
    for child in node.traverse():
        cids = child.get("ids", [])
        if cids:
            return cids[0]
    return None


def _best_title(node: nodes.Element, fallback: str) -> str:
    # Your titles are often "(...)" — keep them.
    title_nodes = list(node.traverse(nodes.title))
    if title_nodes:
        t = title_nodes[0].astext().strip()
        if t:
            return t

    for p in node.traverse(nodes.paragraph):
        txt = p.astext().strip()
        if txt.startswith("(") and txt.endswith(")"):
            return txt

    return fallback


def _toc_order(env) -> list[str]:
    """Best-effort TOC order; fallback to sorted docs."""
    master = getattr(env.config, "master_doc", None) or getattr(env.config, "root_doc", None) or "index"
    includes = getattr(env, "toctree_includes", {})

    if not includes or master not in env.found_docs:
        return sorted(env.found_docs)

    seen = set()
    order: list[str] = []

    def walk(d: str):
        if d in seen:
            return
        seen.add(d)
        order.append(d)
        for ch in includes.get(d, []):
            walk(ch)

    walk(master)
    for d in sorted(env.found_docs):
        if d not in seen:
            order.append(d)
    return order


def _collect_items(app, want_type: str):
    """
    Collect theorem-like blocks.
    Returns list of (docname, target, title)

    'target' is the anchor id used in URLs (e.g. theorem-TotalSizeDist).
    """
    env = app.builder.env
    items: list[tuple[str, str, str]] = []

    for docname in _toc_order(env):
        try:
            dt = env.get_doctree(docname)
        except Exception:
            continue

        for n in dt.traverse():
            if not isinstance(n, nodes.Element):
                continue
            if not _node_looks_like(n, want_type):
                continue

            target = _find_anchor_id(n)
            if not target:
                continue

            title = _best_title(n, fallback=target)
            items.append((docname, target, title))

    return items


_num_re = re.compile(r"\b(\d+(?:\.\d+)*)\b")


def _num_key_from_text(s: str) -> tuple[int, ...]:
    m = _num_re.search(s)
    if not m:
        return (10**18,)
    parts = m.group(1).split(".")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            out.append(10**18)
    return tuple(out)


def _resolve_prf_ref(app, fromdocname: str, target: str) -> nodes.Node:
    """
    Resolve a prf:ref cross-reference NOW, so we get visible text like "Theorem 3.2".
    This must use a real pending_xref node (not a generic nodes.Element).
    """
    env = app.builder.env
    builder = app.builder
    domain = env.get_domain("prf")

    xref = pending_xref(
        "",
        refdomain="prf",
        reftype="ref",
        reftarget=target,
        refexplicit=False,
    )
    # Sphinx expects xref['refdoc'] to exist for some domains/builders
    xref["refdoc"] = fromdocname

    contnode = nodes.literal(text=target)  # placeholder; domain will replace this text

    resolved = domain.resolve_xref(
        env=env,
        fromdocname=fromdocname,
        builder=builder,
        typ="ref",
        target=target,
        node=xref,
        contnode=contnode,
    )

    if resolved is not None:
        return resolved

    # Fallback: direct link (won't show numbering)
    # Try to guess the doc it lives in later; for now at least produce a link.
    ref = nodes.reference("", "", refuri=f"#{target}")
    ref += nodes.literal(text=target)
    return ref


def process_prf_listof_nodes(app, doctree, fromdocname):
    for node in list(doctree.traverse(prf_listof_node)):
        want_type = str(node.get("want_type", "theorem")).lower()
        sort_mode = str(node.get("sort", "number")).lower()
        debug = bool(node.get("debug", False))

        raw_items = _collect_items(app, want_type)

        items = []
        for docname, target, title in raw_items:
            refnode = _resolve_prf_ref(app, fromdocname, target)
            ref_text = refnode.astext().strip()
            num_key = _num_key_from_text(ref_text)
            items.append((docname, target, title, refnode, ref_text, num_key))

        if sort_mode in ("number", "num", "theorem-number"):
            items.sort(key=lambda t: t[5])
        elif sort_mode in ("title", "alpha", "alphabetical"):
            items.sort(key=lambda t: (t[2] or "").lower())
        elif sort_mode in ("none", "doc", "document"):
            pass

        container = nodes.container("")

        if debug:
            debug_lines = [
                "prf_listof debug:",
                f"  want_type = {want_type}",
                f"  sort_mode = {sort_mode}",
                f"  extracted items = {len(items)}",
                "  first 30 (target | resolved_text | title):",
            ] + [f"    {tgt} | {rtext} | {title}" for (_d, tgt, title, _r, rtext, _k) in items[:30]]
            container += nodes.literal_block(text="\n".join(debug_lines))

        bullet = nodes.bullet_list()
        for _docname, _target, title, refnode, _ref_text, _k in items:
            li = nodes.list_item("")
            p = nodes.paragraph("")
            # "Theorem 3.2" (as a link)
            p += refnode
            # Title after it
            if title:
                p += nodes.Text(" ")
                p += nodes.emphasis(text=title)
            li += p
            bullet += li

        container += bullet
        node.replace_self(container)


def setup(app):
    app.add_node(prf_listof_node)
    app.add_directive("prf:listof", PrfListOf)
    app.connect("doctree-resolved", process_prf_listof_nodes)
    return {"version": "1.1", "parallel_read_safe": True, "parallel_write_safe": True}
