from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BOOKS_START = "<!-- BOOKS:START -->"
BOOKS_END = "<!-- BOOKS:END -->"

STORE_PATH = Path("books.json")
PAGE_PATH = Path("books.html")


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _html_to_text(fragment: str) -> str:
    text = fragment
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _name_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z]+", text.casefold())
    return {t for t in tokens if len(t) > 1}


def _looks_like_parenthetical_metadata(value: str) -> bool:
    text = normalize(value)
    if not text:
        return False
    if re.fullmatch(r"\d+", text):
        return True
    metadata_terms = [
        "edition",
        "classics",
        "rediscovered",
        "series",
        "vol",
        "volume",
        "paperback",
        "hardcover",
        "anniversary",
    ]
    return any(term in text for term in metadata_terms)


def _is_placeholder_title(value: str) -> bool:
    text = normalize(value)
    if not text:
        return True
    placeholders = {
        "amazon.com",
        "amazon book",
        "book",
        "books",
        "untitled",
        "untitled book",
    }
    if text in placeholders:
        return True
    if re.fullmatch(r"amazon\.(com|co\.uk|ca|de|fr|it|es|com\.au)", text):
        return True
    return False


def _cleanup_title_and_author(raw_title: str, raw_author: str = "") -> tuple[str, str]:
    title = _clean_whitespace(html.unescape(raw_title or ""))
    author = _clean_whitespace(html.unescape(raw_author or ""))

    if title:
        title = re.sub(r"^\s*Amazon\.com:\s*", "", title, flags=re.IGNORECASE).strip()
        title = re.sub(r"\s*:\s*Books?\s*$", "", title, flags=re.IGNORECASE).strip()

        parts = [p.strip() for p in title.split(":") if p.strip()]
        cleaned_parts: list[str] = []
        author_tokens = _name_tokens(author)
        for part in parts:
            if re.fullmatch(r"(97[89][-\s]?)?\d[\d\s-]{8,}\d", part):
                continue
            if re.fullmatch(r"amazon\.(com|co\.uk|ca|de|fr|it|es|com\.au)", part, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"books?", part, flags=re.IGNORECASE):
                continue
            if part.count(",") >= 2 and author_tokens and len(_name_tokens(part) & author_tokens) >= 1:
                continue
            cleaned_parts.append(part)

        if cleaned_parts:
            if len(cleaned_parts) >= 2:
                subtitle = cleaned_parts[1]
                subtitle_words = len(re.findall(r"[A-Za-z0-9]+", subtitle))
                promo_terms = ["classic", "defined", "bestseller", "award", "complete", "ultimate", "edition"]
                if subtitle_words >= 10 or any(term in normalize(subtitle) for term in promo_terms):
                    cleaned_parts = [cleaned_parts[0]]
            if len(cleaned_parts) >= 2 and "," in cleaned_parts[-1]:
                candidate_author = cleaned_parts[-1]
                if not author:
                    author = candidate_author
                if (
                    not author
                    or normalize(author) in normalize(candidate_author)
                    or normalize(candidate_author) in normalize(author)
                    or _name_tokens(author) == _name_tokens(candidate_author)
                ):
                    cleaned_parts = cleaned_parts[:-1]
            title = ": ".join(cleaned_parts).strip()

    author = re.sub(r"^\s*(by|author)\s+", "", author, flags=re.IGNORECASE).strip()
    author = re.sub(r"\s*Visit Amazon's.*$", "", author, flags=re.IGNORECASE).strip()
    author = re.sub(r"\s*Page$", "", author, flags=re.IGNORECASE).strip()
    if title and author:
        title_parts = [p.strip() for p in title.split(":") if p.strip()]
        if title_parts and _name_tokens(title_parts[-1]) == _name_tokens(author):
            title = ": ".join(title_parts[:-1]).strip()
    if title and author and normalize(title).endswith(normalize(author)):
        title = re.sub(rf"\s*[:\-]\s*{re.escape(author)}\s*$", "", title, flags=re.IGNORECASE).strip()

    # Strip trailing parenthetical metadata such as "(16)" or "(Rediscovered Classics)".
    while True:
        match = re.search(r"\s*\(([^()]*)\)\s*$", title)
        if not match:
            break
        group = match.group(1).strip()
        if not _looks_like_parenthetical_metadata(group):
            break
        title = title[: match.start()].strip()

    return title, author


def infer_section_from_taxonomy(taxonomy: str) -> str:
    text = normalize(taxonomy)
    if not text:
        return "Non-fiction"

    def _has_term(value: str, term: str) -> bool:
        pattern = r"\b" + re.escape(term.casefold()) + r"\b"
        return re.search(pattern, value) is not None

    fiction_terms = [
        "fiction",
        "novel",
        "literature",
        "fantasy",
        "science fiction",
        "mystery",
        "thriller",
        "romance",
        "young adult",
        "classics",
    ]
    nonfiction_terms = [
        "nonfiction",
        "non-fiction",
        "non fiction",
        "politics",
        "social sciences",
        "philosophy",
        "ethics",
        "morality",
        "history",
        "biography",
        "reference",
        "writing",
        "journalism",
        "publishing",
        "business",
        "science",
        "self-help",
        "economics",
        "religion",
    ]

    has_nonfiction = any(_has_term(text, term) for term in nonfiction_terms)
    has_fiction = any(_has_term(text, term) for term in fiction_terms)

    if has_fiction and not has_nonfiction:
        return "Fiction"
    return "Non-fiction"


def load_store(path: Path = STORE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"sections": []}

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError("books.json must be an object with a 'sections' list.")
    return data


def save_store(store: dict[str, Any], path: Path = STORE_PATH) -> None:
    path.write_text(json.dumps(store, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ensure_section(store: dict[str, Any], name: str) -> dict[str, Any]:
    for section in store["sections"]:
        if normalize(str(section.get("name", ""))) == normalize(name):
            if "books" not in section or not isinstance(section["books"], list):
                section["books"] = []
            return section

    section = {"name": name, "books": []}
    store["sections"].append(section)
    return section


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


def _is_amazon_host(host: str) -> bool:
    host = host.casefold()
    return host == "a.co" or host.endswith("amazon.com")


def validate_book_url(url: str) -> tuple[bool, str]:
    if not url or not url.strip():
        return False, "URL is required."

    raw = url.strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return False, "URL must be an http(s) URL."
    if not parsed.netloc:
        return False, "URL host is required."
    if not _is_amazon_host(parsed.netloc):
        return False, "URL must be an Amazon or a.co URL."

    clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return True, clean


def default_title(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
    suffix = suffix.strip()
    if suffix:
        return f"Amazon Book {suffix}"
    return "Untitled Amazon Book"


def fetch_book_title(url: str) -> str | None:
    metadata = fetch_book_metadata(url)
    return metadata.get("title") or None


def fetch_book_metadata(url: str) -> dict[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "interests-site-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            final_url = resp.geturl()
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}

    def _extract_meta(attr: str, value: str) -> str:
        pattern = rf'<meta[^>]+{attr}\s*=\s*["\']{re.escape(value)}["\'][^>]*content\s*=\s*["\'](.*?)["\']'
        m = re.search(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_text(pattern: str) -> str:
        match = re.search(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return _html_to_text(match.group(1))

    title = _extract_meta("property", "og:title") or _extract_meta("name", "twitter:title")
    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1)

    cover_url = _extract_meta("property", "og:image") or _extract_meta("name", "twitter:image")
    author = _extract_meta("name", "author")
    subtitle_text = _extract_text(r'<span[^>]*id=["\']productSubtitle["\'][^>]*>(.*?)</span>')
    description = ""
    rating = ""
    rating_count = ""
    publication_date = ""
    if not author:
        byline_match = re.search(
            r'id=["\']bylineInfo["\'][^>]*>(.*?)</(?:a|span)>', raw_html, flags=re.IGNORECASE | re.DOTALL
        )
        if byline_match:
            author = _strip_tags(byline_match.group(1))

    acr_match = re.search(r'id=["\']acrPopover["\'][^>]*title=["\']([^"\']+)["\']', raw_html, flags=re.IGNORECASE)
    if acr_match:
        rating = _clean_whitespace(html.unescape(acr_match.group(1)))
    rating_count_match = re.search(
        r'id=["\']acrCustomerReviewText["\'][^>]*>\s*([^<]+?)\s*</', raw_html, flags=re.IGNORECASE | re.DOTALL
    )
    if rating_count_match:
        rating_count = _clean_whitespace(html.unescape(rating_count_match.group(1)))

    if subtitle_text:
        date_match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            subtitle_text,
        )
        if date_match:
            publication_date = _clean_whitespace(date_match.group(0))
    if not publication_date:
        pub_match = re.search(r'Publisher[^<]{0,120}\(([^()]*\d{4}[^()]*)\)', raw_html, flags=re.IGNORECASE | re.DOTALL)
        if pub_match:
            publication_date = _clean_whitespace(html.unescape(pub_match.group(1)))

    taxonomy = ""
    crumb_block = re.search(
        r'id=["\']wayfinding-breadcrumbs_feature_div["\'].*?</ul>', raw_html, flags=re.IGNORECASE | re.DOTALL
    )
    if crumb_block:
        anchors = re.findall(r"<a[^>]*>(.*?)</a>", crumb_block.group(0), flags=re.IGNORECASE | re.DOTALL)
        crumbs = [_clean_whitespace(html.unescape(_strip_tags(a))) for a in anchors]
        crumbs = [c for c in crumbs if c]
        if crumbs:
            taxonomy = " > ".join(crumbs)
    if not taxonomy:
        fallback = re.search(r"Books(?:\s*(?:>|&rsaquo;|&#8250;)\s*[^<\"\n]+){1,8}", raw_html, flags=re.IGNORECASE)
        if fallback:
            taxonomy = html.unescape(fallback.group(0))

    description = _extract_text(r'id=["\']bookDescription_feature_div["\'][^>]*>.*?<noscript>\s*(.*?)\s*</noscript>')
    if not description:
        description = _extract_text(
            r'id=["\']bookDescription_feature_div["\'][^>]*>.*?<div[^>]*a-expander-content[^>]*>(.*?)</div>'
        )
    if not description:
        description = _extract_text(r'id=["\']editorialReviews_feature_div["\'][^>]*>(.*?)</(?:div|section)>')
    if not description:
        description = _extract_meta("property", "og:description") or _extract_meta("name", "description")
    description = _clean_whitespace(description)

    title, author = _cleanup_title_and_author(title, author)
    if _is_placeholder_title(title):
        title = ""

    cover_url = cover_url.strip()
    if not cover_url:
        m = re.search(r'id=["\']landingImage["\'][^>]*data-old-hires=["\'](.*?)["\']', raw_html, flags=re.IGNORECASE)
        if m:
            cover_url = m.group(1).strip()
    if not cover_url:
        m = re.search(r'"hiRes"\s*:\s*"([^"]+)"', raw_html, flags=re.IGNORECASE)
        if m:
            cover_url = m.group(1).strip()
    if not cover_url:
        m = re.search(r'"large"\s*:\s*"([^"]+)"', raw_html, flags=re.IGNORECASE)
        if m:
            cover_url = m.group(1).strip()
    if cover_url:
        cover_url = urllib.parse.urljoin(final_url, cover_url)

    return {
        "title": title or "",
        "cover_url": cover_url,
        "author": author,
        "taxonomy": taxonomy,
        "description": description,
        "publication_date": publication_date,
        "rating": rating,
        "rating_count": rating_count,
    }


def upsert_book(section: str, title: str, url: str, dry_run: bool = False) -> dict[str, Any]:
    section_name = str(section or "").strip()
    book_title = str(title or "").strip()
    raw_url = str(url or "").strip()

    ok, validated = validate_book_url(raw_url)
    if not ok:
        return {
            "status": "error",
            "reason": validated,
            "section": section_name,
            "title": book_title,
            "url": raw_url,
        }

    canonical_url = validated

    try:
        metadata = fetch_book_metadata(raw_url or canonical_url)
        metadata_title = str(metadata.get("title", "")).strip()
        metadata_cover_url = str(metadata.get("cover_url", "")).strip()
        metadata_author = str(metadata.get("author", "")).strip()
        cleaned_input_title, metadata_author = _cleanup_title_and_author(book_title, metadata_author)
        cleaned_metadata_title, _ = _cleanup_title_and_author(metadata_title, metadata_author)
        if cleaned_metadata_title and not _is_placeholder_title(cleaned_metadata_title) and (
            not cleaned_input_title or _is_placeholder_title(cleaned_input_title)
        ):
            book_title = cleaned_metadata_title
        elif cleaned_input_title and not _is_placeholder_title(cleaned_input_title):
            book_title = cleaned_input_title
        else:
            book_title = default_title(canonical_url)

        inferred_section = infer_section_from_taxonomy(str(metadata.get("taxonomy", "")))
        if str(metadata.get("taxonomy", "")).strip():
            # Trust taxonomy over extracted section to avoid LLM misclassification.
            section_name = inferred_section
        elif not section_name or normalize(section_name) == "unsorted":
            section_name = inferred_section

        store = load_store(STORE_PATH)
        target_section = ensure_section(store, section_name)

        title_key = normalize(book_title)
        url_key = normalize(canonical_url)

        known_urls = {
            normalize(str(book.get("url", "")))
            for sec in store.get("sections", [])
            for book in sec.get("books", [])
            if isinstance(book, dict)
        }
        section_titles = {
            normalize(str(book.get("title", "")))
            for book in target_section.get("books", [])
            if isinstance(book, dict)
        }

        if title_key in section_titles:
            return {
                "status": "skipped",
                "reason": "Duplicate title in section.",
                "section": target_section.get("name", section_name),
                "title": book_title,
                "url": canonical_url,
            }

        if url_key in known_urls:
            return {
                "status": "skipped",
                "reason": "Duplicate URL in store.",
                "section": target_section.get("name", section_name),
                "title": book_title,
                "url": canonical_url,
            }

        entry: dict[str, Any] = {"title": book_title, "url": canonical_url}
        if metadata_cover_url:
            entry["cover_url"] = metadata_cover_url
        if metadata_author:
            entry["author"] = metadata_author
        target_section["books"].append(entry)

        if not dry_run:
            save_store(store, STORE_PATH)

        return {
            "status": "added",
            "reason": "",
            "section": target_section.get("name", section_name),
            "title": book_title,
            "url": canonical_url,
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
            "section": section_name,
            "title": book_title,
            "url": canonical_url,
        }


def render_books_block(store: dict[str, Any]) -> str:
    sections = [
        section
        for section in store.get("sections", [])
        if isinstance(section, dict)
    ]
    sections = sorted(sections, key=lambda section: str(section.get("name", "")).casefold())

    preferred_order = {"fiction": 0, "non-fiction": 1, "nonfiction": 1}
    sections.sort(key=lambda section: (preferred_order.get(normalize(str(section.get("name", ""))), 99), str(section.get("name", "")).casefold()))

    first_section_name = html.escape(str(sections[0].get("name", "Books"))) if sections else "Books"
    first_section_id = normalize(str(sections[0].get("name", ""))).replace(" ", "-") if sections else "books"
    first_count = len([book for book in sections[0].get("books", []) if isinstance(book, dict)]) if sections else 0
    first_book_payload: dict[str, str] | None = None

    lines: list[str] = ['<div class="books-hub">']
    metadata_cache: dict[str, dict[str, str]] = {}

    lines.append("")
    lines.append('  <aside class="books-rail">')
    lines.append('    <div class="books-rail-copy">')
    lines.append('      <div class="music-kicker">Shelves</div>')
    lines.append('      <h3 class="books-rail-title">Books</h3>')
    lines.append('      <p class="books-rail-caption">Move between fiction and non-fiction from the left rail and browse the shelf in the center.</p>')
    lines.append("    </div>")
    lines.append('    <div class="books-category-tabs" role="tablist" aria-label="Book categories">')

    for section in sections:
        section_name_raw = str(section.get("name", "Untitled"))
        section_name = html.escape(section_name_raw)
        section_id = normalize(section_name_raw).replace(" ", "-")
        books = [book for book in section.get("books", []) if isinstance(book, dict)]
        active_class = " is-active" if section_id == first_section_id else ""
        lines.append(
            f'      <button class="books-category-tab{active_class}" type="button" role="tab"'
            f' aria-selected="{"true" if active_class else "false"}"'
            f' aria-controls="books-panel-{section_id}" data-panel="{section_id}">'
        )
        lines.append(f'        <span class="books-category-name">{section_name}</span>')
        lines.append(f'        <span class="books-category-count">{len(books)}</span>')
        lines.append("      </button>")

    lines.append("    </div>")
    lines.append("  </aside>")
    lines.append("")
    lines.append('  <section class="books-stage">')
    lines.append('    <div class="books-stage-header">')
    lines.append('      <div class="music-kicker">Selected Shelf</div>')
    lines.append(f'      <h3 class="books-stage-title" id="books-active-category">{first_section_name}</h3>')
    lines.append(f'      <div class="books-stage-meta" id="books-active-count">{first_count} book{"s" if first_count != 1 else ""}</div>')
    lines.append('      <p class="books-stage-caption">Each tile links straight to Amazon. Covers stay visible so the page feels like an actual shelf.</p>')
    lines.append("    </div>")
    lines.append("")
    lines.append('    <div class="books-panels">')

    for section in sections:
        section_name_raw = str(section.get("name", "Untitled"))
        section_name = html.escape(section_name_raw)
        section_id = normalize(section_name_raw).replace(" ", "-")
        hidden_attr = "" if section_id == first_section_id else " hidden"
        lines.append(
            f'      <section class="books-category-panel" id="books-panel-{section_id}" data-panel="{section_id}"'
            f' data-section-name="{section_name}" data-book-count="{len([book for book in section.get("books", []) if isinstance(book, dict)])}"{hidden_attr}>'
        )
        lines.append('        <div class="books-card-grid">')

        books = [book for book in section.get("books", []) if isinstance(book, dict)]
        for book in books:
            title_raw = str(book.get("title", "Book")).strip()
            url_raw = str(book.get("url", "")).strip()

            metadata = metadata_cache.get(url_raw)
            if metadata is None:
                metadata = fetch_book_metadata(url_raw) if url_raw else {}
                metadata_cache[url_raw] = metadata

            metadata_title = str(metadata.get("title", "")).strip()
            if metadata_title and (normalize(title_raw).startswith("amazon book ") or normalize(title_raw) == normalize(url_raw.split("/")[-1])):
                title_raw = metadata_title

            author_raw = str(book.get("author", "")).strip() or str(metadata.get("author", "")).strip()
            title_raw, author_raw = _cleanup_title_and_author(title_raw, author_raw)
            cover_url_raw = str(book.get("cover_url", "")).strip() or str(metadata.get("cover_url", "")).strip()
            description_raw = _clean_whitespace(str(metadata.get("description", "")).strip())
            publication_date_raw = _clean_whitespace(str(metadata.get("publication_date", "")).strip())
            rating_raw = _clean_whitespace(str(metadata.get("rating", "")).strip())
            rating_count_raw = _clean_whitespace(str(metadata.get("rating_count", "")).strip())

            title = html.escape(title_raw)
            url = html.escape(url_raw)
            author = html.escape(author_raw)
            if first_book_payload is None:
                first_book_payload = {
                    "title": title_raw,
                    "author": author_raw,
                    "url": url_raw,
                    "description": description_raw,
                    "publication_date": publication_date_raw,
                    "rating": rating_raw,
                    "rating_count": rating_count_raw,
                    "section_name": section_name_raw,
                }
            lines.append(
                '          <button class="book-card'
                f'{" is-active" if first_book_payload and first_book_payload["url"] == url_raw else ""}"'
                f' type="button" data-title="{title}" data-author="{author}" data-url="{url}"'
                f' data-description="{html.escape(description_raw)}"'
                f' data-publication-date="{html.escape(publication_date_raw)}"'
                f' data-rating="{html.escape(rating_raw)}"'
                f' data-rating-count="{html.escape(rating_count_raw)}"'
                f' data-section="{section_name}" aria-label="Show summary for {title}">'
            )
            lines.append('            <div class="book-card-media">')
            if cover_url_raw:
                lines.append(
                    f'              <img class="book-cover" src="{html.escape(cover_url_raw)}" alt="Cover for {title}" loading="lazy">'
                )
            else:
                lines.append('              <div class="book-cover book-cover-placeholder">Book</div>')
            lines.append("            </div>")
            lines.append('            <div class="book-card-body">')
            lines.append(f'              <div class="book-card-title">{title}</div>')
            if author_raw:
                lines.append(f'              <div class="book-author">{author}</div>')
            lines.append('              <div class="book-card-link">Show summary</div>')
            lines.append("            </div>")
            lines.append("          </button>")

        if not books:
            lines.append('          <div class="book-empty">No books yet.</div>')

        lines.append("        </div>")
        lines.append("      </section>")

    lines.append("")
    lines.append("    </div>")
    lines.append("  </section>")
    lines.append("")
    lines.append('  <aside class="books-summary-panel">')
    lines.append('    <div class="books-summary-copy">')
    lines.append('      <div class="music-kicker">Book Summary</div>')
    if first_book_payload:
        default_title = html.escape(first_book_payload["title"] or "Selected book")
        default_author = html.escape(first_book_payload["author"])
        default_url = html.escape(first_book_payload["url"])
        default_description = html.escape(first_book_payload["description"])
        default_pub = html.escape(first_book_payload["publication_date"])
        default_rating = html.escape(first_book_payload["rating"])
        default_rating_count = html.escape(first_book_payload["rating_count"])
        default_section = html.escape(first_book_payload["section_name"])
    else:
        default_title = "No book selected"
        default_author = ""
        default_url = "#"
        default_description = "Pick a book from the shelf to load its Amazon summary here."
        default_pub = ""
        default_rating = ""
        default_rating_count = ""
        default_section = "Books"
    lines.append(f'      <h3 class="books-summary-section" id="books-summary-section">{default_section}</h3>')
    lines.append(f'      <div class="books-summary-title" id="books-summary-title">{default_title}</div>')
    lines.append(f'      <div class="books-summary-author" id="books-summary-author">{default_author}</div>')
    lines.append('      <div class="books-summary-meta">')
    lines.append(f'        <div class="books-summary-pill" id="books-summary-date">{default_pub}</div>')
    lines.append(f'        <div class="books-summary-pill" id="books-summary-rating">{default_rating}</div>')
    lines.append(f'        <div class="books-summary-pill" id="books-summary-rating-count">{default_rating_count}</div>')
    lines.append("      </div>")
    lines.append(
        '      <a class="books-summary-link inline-link" id="books-summary-link" '
        f'href="{default_url}" target="_blank" rel="noopener noreferrer">Open on Amazon</a>'
    )
    lines.append("    </div>")
    lines.append(f'    <div class="books-summary-body" id="books-summary-body">{default_description}</div>')
    lines.append("  </aside>")
    lines.append("</div>")
    return "\n".join(lines)


def render_site(dry_run: bool = False) -> dict[str, Any]:
    try:
        store = load_store(STORE_PATH)
        html_text = PAGE_PATH.read_text(encoding="utf-8")

        pattern = re.compile(rf"({re.escape(BOOKS_START)})(.*?)({re.escape(BOOKS_END)})", re.DOTALL)
        match = pattern.search(html_text)
        if not match:
            return {
                "status": "error",
                "reason": "books.html is missing BOOKS markers: <!-- BOOKS:START --> and <!-- BOOKS:END -->",
                "changed": False,
            }

        books_html = render_books_block(store)
        updated_html = html_text[: match.start(2)] + "\n" + books_html + "\n" + html_text[match.end(2) :]

        changed = updated_html != html_text
        if changed and not dry_run:
            PAGE_PATH.write_text(updated_html, encoding="utf-8")

        return {"status": "ok", "reason": "", "changed": changed}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "changed": False}


def refresh_store_metadata(dry_run: bool = False) -> dict[str, Any]:
    try:
        store = load_store(STORE_PATH)
        refreshed: dict[str, Any] = {"sections": [{"name": "Fiction", "books": []}, {"name": "Non-fiction", "books": []}]}
        seen_urls: set[str] = set()

        for section in store.get("sections", []):
            for book in section.get("books", []):
                if not isinstance(book, dict):
                    continue
                raw_url = str(book.get("url", "")).strip()
                ok, canonical_or_reason = validate_book_url(raw_url)
                if not ok:
                    continue
                canonical_url = canonical_or_reason
                url_key = normalize(canonical_url)
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)

                metadata = fetch_book_metadata(canonical_url)
                metadata_author = str(book.get("author", "")).strip() or str(metadata.get("author", "")).strip()
                metadata_title = str(metadata.get("title", "")).strip()
                stored_title = str(book.get("title", "")).strip()
                cleaned_metadata_title, metadata_author = _cleanup_title_and_author(metadata_title, metadata_author)
                cleaned_stored_title, metadata_author = _cleanup_title_and_author(stored_title, metadata_author)
                if cleaned_metadata_title and not _is_placeholder_title(cleaned_metadata_title):
                    title = cleaned_metadata_title
                elif cleaned_stored_title and not _is_placeholder_title(cleaned_stored_title):
                    title = cleaned_stored_title
                else:
                    title = default_title(canonical_url)
                author = metadata_author
                taxonomy = str(metadata.get("taxonomy", "")).strip()
                section_name = infer_section_from_taxonomy(taxonomy) if taxonomy else str(section.get("name", "")).strip() or "Non-fiction"
                target = ensure_section(refreshed, section_name)
                entry: dict[str, Any] = {"title": title, "url": canonical_url}
                cover = str(book.get("cover_url", "")).strip() or str(metadata.get("cover_url", "")).strip()
                if cover:
                    entry["cover_url"] = cover
                if author:
                    entry["author"] = author
                target["books"].append(entry)

        if not dry_run:
            save_store(refreshed, STORE_PATH)

        return {"status": "ok", "reason": "", "changed": True}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "changed": False}
