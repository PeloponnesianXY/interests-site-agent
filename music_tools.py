from __future__ import annotations

from typing import Any

import music_ops

AddSongResult = dict[str, Any]
RenderResult = dict[str, Any]


class SectionCatalogTool:
    def list_sections(self) -> list[str]:
        return music_ops.get_section_names()


class SpotifyUrlNormalizerTool:
    def normalize(self, url: str) -> str:
        ok, detail = music_ops.validate_spotify_embed_url(url)
        if not ok:
            raise ValueError(detail)
        return detail


class SongAdderTool:
    def add_song(self, section: str, title: str, embed_url: str, dry_run: bool = False) -> AddSongResult:
        return music_ops.upsert_song(section=section, title=title, url=embed_url, dry_run=dry_run)


class SiteRendererTool:
    def render(self, dry_run: bool = False) -> RenderResult:
        return music_ops.render_site(dry_run=dry_run)
