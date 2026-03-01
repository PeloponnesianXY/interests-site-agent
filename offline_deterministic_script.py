from __future__ import annotations

import argparse
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda


MUSIC_START = "<!-- MUSIC:START -->"
MUSIC_END = "<!-- MUSIC:END -->"

SPOTIFY_EMBED_RE = re.compile(
    r"^https://open\.spotify\.com/embed/(?P<kind>track|album|playlist)/(?P<item_id>[A-Za-z0-9]+)(?:\?.*)?$"
)
SPOTIFY_SHARE_RE = re.compile(
    r"^https://open\.spotify\.com/(?P<kind>track|album|playlist)/(?P<item_id>[A-Za-z0-9]+)(?:\?.*)?$"
)
SPOTIFY_URI_RE = re.compile(r"^spotify:(?P<kind>track|album|playlist):(?P<item_id>[A-Za-z0-9]+)$")


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def parse_spotify_url(raw_url: str) -> tuple[str, str, str] | None:
    text = raw_url.strip()

    for pattern in (SPOTIFY_EMBED_RE, SPOTIFY_SHARE_RE, SPOTIFY_URI_RE):
        match = pattern.fullmatch(text)
        if match:
            kind = match.group("kind")
            item_id = match.group("item_id")
            embed_url = f"https://open.spotify.com/embed/{kind}/{item_id}?utm_source=generator"
            return kind, item_id, embed_url

    return None


def default_title(kind: str, item_id: str) -> str:
    return f"Spotify {kind.title()} {item_id[:8]}"


def fetch_spotify_title(kind: str, item_id: str) -> str | None:
    share_url = f"https://open.spotify.com/{kind}/{item_id}"
    endpoint = "https://open.spotify.com/oembed?url=" + urllib.parse.quote(share_url, safe="")
    req = urllib.request.Request(endpoint, headers={"User-Agent": "music-site-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    title = str(data.get("title", "")).strip()
    if not title:
        return None
    return title.replace("|", "-")


def load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sections": []}

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError("music.json must be an object with a 'sections' list.")
    return data


def parse_inbox(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 2:
            section, raw_url = parts
            title = ""
            had_title = False
        elif len(parts) == 3:
            section, title, raw_url = parts
            had_title = True
        else:
            entries.append(
                {
                    "line_no": line_no,
                    "section": "Uncategorized",
                    "error": "Expected: Section | URL OR Section | Title | URL",
                }
            )
            continue

        if not section or not raw_url:
            entries.append(
                {
                    "line_no": line_no,
                    "section": section or "Uncategorized",
                    "error": "Section and URL are required.",
                }
            )
            continue

        parsed = parse_spotify_url(raw_url)
        if not parsed:
            entries.append(
                {
                    "line_no": line_no,
                    "section": section,
                    "error": "URL must match spotify track/album/playlist URL, embed URL, or URI.",
                }
            )
            continue

        kind, item_id, embed_url = parsed
        final_title = title if title else (fetch_spotify_title(kind, item_id) or default_title(kind, item_id))

        entries.append(
            {
                "line_no": line_no,
                "section": section,
                "title": final_title,
                "url": embed_url,
                "input_url": raw_url,
                "had_title": had_title,
            }
        )

    return entries


def ensure_section(store: dict[str, Any], name: str) -> dict[str, Any]:
    for section in store["sections"]:
        if normalize(section.get("name", "")) == normalize(name):
            if "songs" not in section or not isinstance(section["songs"], list):
                section["songs"] = []
            return section

    section = {"name": name, "songs": []}
    store["sections"].append(section)
    return section


def upsert_store(state: dict[str, Any]) -> dict[str, Any]:
    store: dict[str, Any] = state["store"]
    entries: list[dict[str, Any]] = state["entries"]

    summary: dict[str, dict[str, int]] = {}

    def bump(section: str, key: str) -> None:
        if section not in summary:
            summary[section] = {"added": 0, "skipped": 0, "errors": 0}
        summary[section][key] += 1

    known_urls = {
        normalize(song.get("url", ""))
        for section in store.get("sections", [])
        for song in section.get("songs", [])
        if isinstance(song, dict)
    }

    for item in entries:
        section_name = item.get("section", "Uncategorized")

        if "error" in item:
            bump(section_name, "errors")
            continue

        section = ensure_section(store, section_name)

        title_key = normalize(item["title"])
        url_key = normalize(item["url"])

        section_titles = {normalize(song.get("title", "")) for song in section["songs"] if isinstance(song, dict)}

        if title_key in section_titles or url_key in known_urls:
            bump(section_name, "skipped")
            continue

        section["songs"].append({"title": item["title"], "url": item["url"]})
        known_urls.add(url_key)
        bump(section_name, "added")

    state["store"] = store
    state["summary"] = summary
    return state


def render_music_block(state: dict[str, Any]) -> dict[str, Any]:
    store: dict[str, Any] = state["store"]

    lines: list[str] = ["<div class=\"styles-grid\">"]

    for section in store.get("sections", []):
        section_name = html.escape(section.get("name", "Untitled"))
        lines.append("")
        lines.append("  <div class=\"style-tile\">")
        lines.append(f"    <h3>{section_name}</h3>")
        lines.append("")
        lines.append("    <div class=\"style-embed\">")

        for song in section.get("songs", []):
            title = html.escape(song.get("title", "Spotify Embed"))
            url = html.escape(song.get("url", ""))
            lines.append("      <iframe class=\"spotify\"")
            lines.append(f"        title=\"{title}\"")
            lines.append(f"        src=\"{url}\"")
            lines.append("        width=\"100%\" height=\"152\" frameborder=\"0\"")
            lines.append(
                "        allow=\"autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture\""
            )
            lines.append("        loading=\"lazy\" scrolling=\"no\"></iframe>")
            lines.append("")

        lines.append("    </div>")
        lines.append("  </div>")

    lines.append("")
    lines.append("</div>")

    state["music_html"] = "\n".join(lines)
    return state


def write_outputs(state: dict[str, Any]) -> dict[str, Any]:
    page_path: Path = state["page_path"]
    store_path: Path = state["store_path"]

    html_text = page_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"({re.escape(MUSIC_START)})(.*?)({re.escape(MUSIC_END)})", re.DOTALL)
    match = pattern.search(html_text)

    if not match:
        raise ValueError("index.html is missing MUSIC markers: <!-- MUSIC:START --> and <!-- MUSIC:END -->")

    updated_html = html_text[: match.start(2)] + "\n" + state["music_html"] + "\n" + html_text[match.end(2) :]
    page_path.write_text(updated_html, encoding="utf-8")

    store_path.write_text(json.dumps(state["store"], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return state


def rewrite_inbox_with_titles(state: dict[str, Any]) -> dict[str, Any]:
    inbox_path: Path = state["inbox_path"]
    original_lines = inbox_path.read_text(encoding="utf-8").splitlines()
    by_line = {entry["line_no"]: entry for entry in state["entries"] if "error" not in entry}

    new_lines: list[str] = []
    for i, raw in enumerate(original_lines, start=1):
        entry = by_line.get(i)
        if not entry or entry.get("had_title"):
            new_lines.append(raw)
            continue

        section = entry["section"]
        title = entry["title"]
        input_url = entry["input_url"]
        new_lines.append(f"{section} | {title} | {input_url}")

    inbox_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return state


def print_summary(state: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, dict[str, int]] = state["summary"]

    total_added = 0
    total_skipped = 0
    total_errors = 0

    for section in sorted(summary.keys(), key=str.casefold):
        added = summary[section]["added"]
        skipped = summary[section]["skipped"]
        errors = summary[section]["errors"]

        total_added += added
        total_skipped += skipped
        total_errors += errors

        print(f"{section}: added {added}, skipped {skipped}, errors {errors}")

    print(f"Done. Added {total_added}, skipped {total_skipped}, errors {total_errors}.")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline music automation: inbox -> music.json -> index.html block render."
    )
    parser.add_argument("--inbox", default="songs_to_add.txt", help="Path to inbox text file")
    parser.add_argument("--page", default="index.html", help="Path to music HTML page")
    parser.add_argument("--store", default="music.json", help="Path to canonical JSON store")
    args = parser.parse_args()

    inbox_path = Path(args.inbox)
    page_path = Path(args.page)
    store_path = Path(args.store)

    if not inbox_path.exists():
        raise SystemExit(f"Inbox file not found: {inbox_path}")
    if not page_path.exists():
        raise SystemExit(f"Music page not found: {page_path}")

    state: dict[str, Any] = {
        "entries": parse_inbox(inbox_path),
        "store": load_store(store_path),
        "page_path": page_path,
        "store_path": store_path,
        "inbox_path": inbox_path,
    }

    pipeline = (
        RunnableLambda(upsert_store)
        | RunnableLambda(render_music_block)
        | RunnableLambda(write_outputs)
        | RunnableLambda(rewrite_inbox_with_titles)
        | RunnableLambda(print_summary)
    )

    pipeline.invoke(state)


if __name__ == "__main__":
    main()
