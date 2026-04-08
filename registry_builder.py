"""
registry_builder.py
--------------------
Scrapes DOCS_URL (Cisco NX-OS show-command pages) and produces a structured
command_registry.json where every entry represents one NX-OS CLI command
with its syntax, description, command mode, usage guidelines, and examples.

The registry is the configurable intermediate layer between raw HTML and the
FAISS index.  Edit it freely:
  - Set "enabled": false to exclude a command from the index.
  - Add or override "tags" for better retrieval.
  - Manually add commands that weren't scraped.

Re-run build_registry.py to refresh from live docs; manual edits to
"enabled" and "tags" are preserved across rebuilds (merge logic below).
"""

import json
import os
import re
from urllib.parse import urljoin, urlparse

from config import settings


def _registry_path() -> str:
    return os.path.join(settings.command_registry_dir, "command_registry.json")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def registry_exists() -> bool:
    return os.path.exists(_registry_path())


def build_registry() -> int:
    """
    Crawl DOCS_URL, parse each show-command page into structured entries,
    and save to command_registry/command_registry.json.

    If a registry already exists, manual edits to 'enabled' and 'tags'
    are preserved for entries whose 'id' matches (merge by ID).

    Returns the number of entries written.
    """
    import requests  # imported here so the module loads without requests installed

    headers = {"User-Agent": settings.user_agent}
    entries = _crawl_and_parse(settings.docs_url, headers, settings.docs_max_pages)

    os.makedirs(settings.command_registry_dir, exist_ok=True)

    # Preserve manual edits from an existing registry
    existing_by_id: dict[str, dict] = {}
    if registry_exists():
        with open(_registry_path(), encoding="utf-8") as f:
            for e in json.load(f):
                existing_by_id[e["id"]] = e

    merged: list[dict] = []
    for entry in entries:
        eid = entry["id"]
        if eid in existing_by_id:
            prev = existing_by_id[eid]
            # Preserve only manually-curated fields
            entry["enabled"] = prev.get("enabled", True)
            if prev.get("tags"):
                entry["tags"] = prev["tags"]
        merged.append(entry)

    with open(_registry_path(), "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"[registry] Wrote {len(merged)} entries → {_registry_path()}")
    return len(merged)


def load_registry() -> list[dict]:
    """
    Load command_registry.json and return only enabled entries.
    Raises FileNotFoundError if the registry hasn't been built yet.
    """
    with open(_registry_path(), encoding="utf-8") as f:
        entries = json.load(f)
    enabled = [e for e in entries if e.get("enabled", True)]
    disabled = len(entries) - len(enabled)
    print(f"[registry] Loaded {len(enabled)} enabled entries"
          + (f" ({disabled} disabled)" if disabled else ""))
    return enabled


def chunks_from_registry(entries: list[dict]) -> list[str]:
    """
    Convert enabled registry entries into rich text chunks for embedding.

    Each chunk contains the full command context so the embedder and
    semantic search have maximum signal per chunk.
    """
    chunks = []
    for entry in entries:
        parts: list[str] = []

        parts.append(f"Command: {entry['command']}")

        if entry.get("syntax"):
            parts.append(f"Syntax: {entry['syntax']}")

        if entry.get("mode"):
            parts.append(f"Mode: {entry['mode']}")

        if entry.get("description"):
            parts.append(f"Description: {entry['description']}")

        if entry.get("usage_guidelines"):
            parts.append(f"Usage Guidelines: {entry['usage_guidelines']}")

        if entry.get("examples"):
            ex_lines = "\n  ".join(entry["examples"][:6])
            parts.append(f"Examples:\n  {ex_lines}")

        if entry.get("tags"):
            parts.append(f"Tags: {', '.join(entry['tags'])}")

        if entry.get("source_url"):
            parts.append(f"Source: {entry['source_url']}")

        chunks.append("\n".join(parts))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Crawl
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_url(raw_url: str) -> str:
    return raw_url.split("#", 1)[0].split("?", 1)[0].strip()


def _is_relevant_show_doc(seed_url: str, candidate_url: str) -> bool:
    seed = urlparse(seed_url)
    cand = urlparse(candidate_url)
    if cand.scheme not in ("http", "https"):
        return False
    if cand.netloc != seed.netloc:
        return False
    if "/command-reference/show/" not in cand.path:
        return False
    if not cand.path.endswith(".html"):
        return False
    return True


def _crawl_and_parse(seed_url: str, headers: dict, max_pages: int) -> list[dict]:
    import time
    import requests
    from bs4 import BeautifulSoup

    MAX_RETRIES   = 3
    RETRY_DELAYS  = [5, 15, 30]   # seconds between retries on 403/429

    visited: set[str] = set()
    failed:  set[str] = set()     # pages that exhausted all retries
    queue:   list[str] = [_normalize_url(seed_url)]
    entries: list[dict] = []

    def _fetch(url: str):
        """Fetch url with retry/back-off on 403/429. Returns Response or None."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, headers=headers, timeout=60)
                if resp.status_code in (403, 429) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"[registry]   {resp.status_code} — retrying in {delay}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES}) ...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"[registry]   error ({exc}) — retrying in {delay}s ...")
                    time.sleep(delay)
                else:
                    print(f"[registry] FAILED after {MAX_RETRIES} attempts: {url} — {exc}")
        return None

    while queue and len(visited) < max_pages:
        current = queue.pop(0)
        if current in visited or current in failed:
            continue

        print(f"[registry] Fetching: {current}")
        resp = _fetch(current)
        if resp is None:
            failed.add(current)
            continue

        visited.add(current)
        soup = BeautifulSoup(resp.text, "html.parser")

        page_entries = _parse_command_page(soup, current)
        if page_entries:
            entries.extend(page_entries)
            print(f"[registry]   → {len(page_entries)} command(s) parsed")
        else:
            print(f"[registry]   → (no commands found on page)")

        for a in soup.find_all("a", href=True):
            absolute = _normalize_url(urljoin(current, a["href"]))
            if absolute not in visited and absolute not in failed and absolute not in queue:
                if _is_relevant_show_doc(seed_url, absolute):
                    queue.append(absolute)

    if failed:
        print(f"[registry] {len(failed)} page(s) permanently failed (add manually if needed):")
        for url in sorted(failed):
            print(f"[registry]   MISSED: {url}")

    print(f"[registry] Done. Crawled {len(visited)} pages, {len(entries)} total entries.")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Parse a single page → list of command entries
# ─────────────────────────────────────────────────────────────────────────────

def _make_id(command: str) -> str:
    """Stable slug ID from command text (e.g. 'show bgp neighbors' → 'show-bgp-neighbors')."""
    return re.sub(r"[^a-z0-9]+", "-", command.lower()).strip("-")


def _auto_tags(command: str) -> list[str]:
    stopwords = {"show", "all", "the", "a", "an", "in", "of", "for", "to", "and"}
    words = re.findall(r"[a-z0-9]+", command.lower())
    return [w for w in words if w not in stopwords and len(w) > 2]


_SECTION_KEYWORDS = {
    "syntax description":  "syntax_desc",
    "syntax":              "syntax",
    "command modes":       "mode",
    "command mode":        "mode",
    "command default":     "default",
    "defaults":            "default",
    "usage guidelines":    "usage",
    "usage guideline":     "usage",
    "usage notes":         "usage",
    "examples":            "examples",
    "example":             "examples",
    "related commands":    "stop",
    "related command":     "stop",
}


def _classify_heading(text: str) -> str | None:
    """Return a section key if the heading matches a known Cisco doc section."""
    t = text.lower().strip()
    for keyword, key in _SECTION_KEYWORDS.items():
        if keyword in t:
            return key
    return None


def _parse_command_page(soup, source_url: str) -> list[dict]:
    """
    Parse one Cisco NX-OS show-command HTML page into structured entries.

    NX-OS command-reference pages follow a predictable template:
      <h1/h2>  show bgp neighbors                   ← command name
      <pre>    show bgp [vrf ...] neighbors [...]    ← syntax block
      Syntax Description (table)
      Command Modes                                  ← EXEC / privileged EXEC
      Usage Guidelines                               ← prose
      Examples                                       ← <pre> code blocks

    Some pages document multiple sub-variants under separate <h2> headings.
    """
    # Strip noise elements
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    entries: list[dict] = []

    # Collect all content elements in document order for a linear scan.
    # Include th/td so we can detect section labels that Cisco puts in table
    # headers (e.g. "Syntax Description", "Command Mode") and read mode values.
    body = soup.find("body") or soup
    all_elements = list(body.find_all(
        ["h1", "h2", "h3", "h4", "h5", "th", "td", "p", "pre", "ul", "ol", "table", "div"],
        recursive=True,
    ))

    # Identify command heading positions
    command_heading_indices: list[int] = []
    for i, el in enumerate(all_elements):
        if el.name in ("h1", "h2", "h3"):
            text = el.get_text(strip=True)
            if re.match(r"show\s+\S", text, re.IGNORECASE):
                command_heading_indices.append(i)

    # Fallback: try page title
    if not command_heading_indices:
        title_el = soup.find("title")
        if title_el:
            m = re.search(r"(show\s+[\w][\w\s/-]*)", title_el.get_text(), re.IGNORECASE)
            if m:
                cmd_text = " ".join(m.group(1).split())
                entries.append(_empty_entry(cmd_text, source_url))
                _fill_entry_from_elements(entries[-1], all_elements, 0, len(all_elements))
        return entries

    # Parse each command section
    for pos, start_idx in enumerate(command_heading_indices):
        end_idx = (
            command_heading_indices[pos + 1]
            if pos + 1 < len(command_heading_indices)
            else len(all_elements)
        )
        heading_el = all_elements[start_idx]
        cmd_text = " ".join(heading_el.get_text(strip=True).split())

        entry = _empty_entry(cmd_text, source_url)
        _fill_entry_from_elements(entry, all_elements, start_idx + 1, end_idx)
        entries.append(entry)

    return entries


def _empty_entry(command: str, source_url: str) -> dict:
    return {
        "id":               _make_id(command),
        "command":          command,
        "syntax":           "",
        "description":      "",
        "mode":             "",
        "usage_guidelines": "",
        "examples":         [],
        "source_url":       source_url,
        "platform":         "NX-OS",
        "enabled":          True,
        "tags":             _auto_tags(command),
    }


def _fill_entry_from_elements(
    entry: dict,
    elements: list,
    start: int,
    end: int,
) -> None:
    """
    Scan elements[start:end] and populate entry fields.

    Cisco NX-OS pages label sections with <p> or <th> text (not always <h3>):
      [p]   show aaa authorization [ all ]  ← syntax (paragraph starting with "show ")
      [p/th] Syntax Description             ← section marker → skip table content
      [p/th] Command Mode                   ← section marker
      [td]   /exec                          ← mode value
      [p/th] Usage Guidelines               ← section marker
      [p]    prose description...
      [p/th] Examples
      [pre]  switch# show ...
    """
    section = "preamble"       # haven't seen the syntax line yet
    description_parts: list[str] = []
    usage_parts: list[str] = []

    for el in elements[start:end]:
        tag = el.name
        raw_text = el.get_text(separator=" ", strip=True)
        text = " ".join(raw_text.split())
        if not text:
            continue

        # ── Section detection: works for any tag (h*, p, th, td, div) ─────
        # Cisco pages use <p> and <th> for section labels, not always <h3>.
        classified = _classify_heading(text)
        if classified == "stop":
            break
        if classified and tag in ("h1", "h2", "h3", "h4", "h5", "th", "p", "div"):
            # Only switch section if the element's ENTIRE text is the keyword.
            # This avoids treating a paragraph that merely mentions "examples"
            # mid-sentence as a section break.
            if len(text) < 60:
                section = classified
                continue

        # ── <pre>/<code>: syntax block or example ─────────────────────────
        if tag in ("pre", "code"):
            if not entry["syntax"] and section in ("preamble", "syntax"):
                entry["syntax"] = text
                section = "post_syntax"
            elif section == "examples":
                for line in raw_text.splitlines():
                    line = line.strip()
                    if line:
                        entry["examples"].append(line)
            continue

        # ── Route by current section ──────────────────────────────────────

        if section == "preamble":
            # First <p> starting with "show " is the syntax line
            if tag in ("p", "li") and text.lower().startswith("show "):
                entry["syntax"] = text
                section = "post_syntax"
            # Anything else before syntax → ignore (it's nav/intro text)

        elif section == "post_syntax":
            # Collect short prose descriptions (not table param rows)
            if tag in ("p", "li") and len(text) > 20 and len(description_parts) < 2:
                description_parts.append(text)

        elif section == "syntax_desc":
            pass  # intentionally skip parameter table rows

        elif section == "mode":
            # Mode value comes from <td>, <p>, or <li> — take the first short one
            if not entry["mode"] and tag in ("td", "p", "li") and len(text) < 80:
                entry["mode"] = _clean_mode_text(text)

        elif section == "usage":
            if tag in ("p", "li") and len(text) > 20:
                usage_parts.append(text)

        elif section == "examples":
            pass  # examples handled above in <pre> branch

    if description_parts:
        entry["description"] = " ".join(description_parts)
    if usage_parts:
        entry["usage_guidelines"] = " ".join(usage_parts[:3])


# ─────────────────────────────────────────────────────────────────────────────
# Small text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_prompt_line(line: str) -> bool:
    """True for shell prompt lines like 'switch# ' or 'switch(config)# '."""
    return bool(re.match(r"^[\w()]+[#>]\s*$", line))


def _clean_mode_text(text: str) -> str:
    """Keep only the mode name portion, drop trailing noise."""
    # e.g. "EXEC mode" → "EXEC", "Privileged EXEC (config)#" → "Privileged EXEC"
    m = re.match(r"([A-Za-z][A-Za-z\s\-]*?)(?:\s+mode|\s*\(|\s*#|$)", text)
    return m.group(1).strip() if m else text.strip()
