from __future__ import annotations

from typing import Any

import books_ops

AddBookResult = dict[str, Any]
RenderResult = dict[str, Any]


class BookSectionCatalogTool:
    def list_sections(self) -> list[str]:
        return books_ops.get_section_names()


class AmazonUrlNormalizerTool:
    def normalize(self, url: str) -> str:
        ok, detail = books_ops.validate_book_url(url)
        if not ok:
            raise ValueError(detail)
        return detail


class BookAdderTool:
    def add_book(self, section: str, title: str, book_url: str, dry_run: bool = False) -> AddBookResult:
        return books_ops.upsert_book(section=section, title=title, url=book_url, dry_run=dry_run)


class BookSiteRendererTool:
    def render(self, dry_run: bool = False) -> RenderResult:
        return books_ops.render_site(dry_run=dry_run)
