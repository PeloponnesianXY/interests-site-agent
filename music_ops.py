from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MUSIC_START = "<!-- MUSIC:START -->"
MUSIC_END = "<!-- MUSIC:END -->"

STORE_PATH = Path("music.json")
PAGE_PATH = Path("index.html")

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


def load_store(path: Path = STORE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"sections": []}

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError("music.json must be an object with a 'sections' list.")
    return data


def save_store(store: dict[str, Any], path: Path = STORE_PATH) -> None:
    path.write_text(json.dumps(store, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ensure_section(store: dict[str, Any], name: str) -> dict[str, Any]:
    for section in store["sections"]:
        if normalize(str(section.get("name", ""))) == normalize(name):
            if "songs" not in section or not isinstance(section["songs"], list):
                section["songs"] = []
            return section

    section = {"name": name, "songs": []}
    store["sections"].append(section)
    return section


def render_music_block(store: dict[str, Any]) -> str:
    lines: list[str] = ['<div class="styles-grid">']

    for section in store.get("sections", []):
        section_name = html.escape(str(section.get("name", "Untitled")))
        lines.append("")
        lines.append('  <div class="style-tile">')
        lines.append(f"    <h3>{section_name}</h3>")
        lines.append("")
        lines.append('    <div class="style-embed">')

        for song in section.get("songs", []):
            title = html.escape(str(song.get("title", "Spotify Embed")))
            url = html.escape(str(song.get("url", "")))
            lines.append(
                f'      <div class="spotify-shell" data-src="{url}" data-title="{title}" aria-label="{title}"></div>'
            )
            lines.append("")

        lines.append("    </div>")
        lines.append("  </div>")

    lines.append("")
    lines.append("</div>")
    return "\n".join(lines)


def get_section_names() -> list[str]:
    try:
        store = load_store(STORE_PATH)
    except Exception:
        return []

    names: list[str] = []
    for section in store.get("sections", []):
        name = str(section.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def validate_spotify_embed_url(url: str) -> tuple[bool, str]:
    if not url or not url.strip():
        return False, "URL is required."

    parsed = parse_spotify_url(url)
    if not parsed:
        return (
            False,
            "URL must match spotify track/album/playlist URL, embed URL, or URI.",
        )

    _kind, _item_id, embed_url = parsed
    return True, embed_url


def upsert_song(section: str, title: str, url: str, dry_run: bool = False) -> dict[str, Any]:
    section_name = str(section or "").strip()
    song_title = str(title or "").strip()
    raw_url = str(url or "").strip()

    if not section_name:
        return {
            "status": "error",
            "reason": "Section is required.",
            "section": section_name,
            "title": song_title,
            "url": raw_url,
        }

    if not song_title:
        return {
            "status": "error",
            "reason": "Title is required.",
            "section": section_name,
            "title": song_title,
            "url": raw_url,
        }

    ok, validated = validate_spotify_embed_url(raw_url)
    if not ok:
        return {
            "status": "error",
            "reason": validated,
            "section": section_name,
            "title": song_title,
            "url": raw_url,
        }

    canonical_url = validated

    try:
        store = load_store(STORE_PATH)
        target_section = ensure_section(store, section_name)

        title_key = normalize(song_title)
        url_key = normalize(canonical_url)

        known_urls = {
            normalize(str(song.get("url", "")))
            for sec in store.get("sections", [])
            for song in sec.get("songs", [])
            if isinstance(song, dict)
        }
        section_titles = {
            normalize(str(song.get("title", "")))
            for song in target_section.get("songs", [])
            if isinstance(song, dict)
        }

        if title_key in section_titles:
            return {
                "status": "skipped",
                "reason": "Duplicate title in section.",
                "section": target_section.get("name", section_name),
                "title": song_title,
                "url": canonical_url,
            }

        if url_key in known_urls:
            return {
                "status": "skipped",
                "reason": "Duplicate URL in store.",
                "section": target_section.get("name", section_name),
                "title": song_title,
                "url": canonical_url,
            }

        target_section["songs"].append({"title": song_title, "url": canonical_url})

        if not dry_run:
            save_store(store, STORE_PATH)

        return {
            "status": "added",
            "reason": "",
            "section": target_section.get("name", section_name),
            "title": song_title,
            "url": canonical_url,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "section": section_name,
            "title": song_title,
            "url": canonical_url,
        }


def render_site(dry_run: bool = False) -> dict[str, Any]:
    try:
        store = load_store(STORE_PATH)
        html_text = PAGE_PATH.read_text(encoding="utf-8")

        pattern = re.compile(rf"({re.escape(MUSIC_START)})(.*?)({re.escape(MUSIC_END)})", re.DOTALL)
        match = pattern.search(html_text)
        if not match:
            return {
                "status": "error",
                "reason": "index.html is missing MUSIC markers: <!-- MUSIC:START --> and <!-- MUSIC:END -->",
                "changed": False,
            }

        music_html = render_music_block(store)
        updated_html = html_text[: match.start(2)] + "\n" + music_html + "\n" + html_text[match.end(2) :]

        changed = updated_html != html_text
        if changed and not dry_run:
            PAGE_PATH.write_text(updated_html, encoding="utf-8")

        return {"status": "ok", "reason": "", "changed": changed}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "changed": False}
