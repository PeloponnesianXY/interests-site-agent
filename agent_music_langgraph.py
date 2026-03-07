from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import books_ops
import music_ops
from books_tools import AmazonUrlNormalizerTool, BookAdderTool, BookSectionCatalogTool, BookSiteRendererTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from music_tools import SectionCatalogTool, SiteRendererTool, SongAdderTool, SpotifyUrlNormalizerTool


SECTION_CATALOG_TOOL = SectionCatalogTool()
SPOTIFY_URL_NORMALIZER_TOOL = SpotifyUrlNormalizerTool()
SONG_ADDER_TOOL = SongAdderTool()
SITE_RENDERER_TOOL = SiteRendererTool()
BOOK_SECTION_CATALOG_TOOL = BookSectionCatalogTool()
AMAZON_URL_NORMALIZER_TOOL = AmazonUrlNormalizerTool()
BOOK_ADDER_TOOL = BookAdderTool()
BOOK_SITE_RENDERER_TOOL = BookSiteRendererTool()


class ExtractedItem(BaseModel):
    section: str = Field(default="")
    title: str = Field(default="")
    url: str = Field(default="")


class ExtractionPayload(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    request_text: str
    extracted_items: list[dict[str, str]]
    valid_items: list[dict[str, Any]]
    invalid_items: list[dict[str, Any]]
    results: list[dict[str, Any]]
    render_result: dict[str, Any]
    model_name: str
    dry_run: bool
    extract_error: str
    token_usage: dict[str, int]


def _make_extraction_messages(
    request_text: str, music_sections: list[str], book_sections: list[str]
) -> list[tuple[str, str]]:
    music_csv = ", ".join(music_sections) if music_sections else "(none)"
    books_csv = ", ".join(book_sections) if book_sections else "(none)"
    system_prompt = (
        "You are an extraction engine. Output STRICT JSON only. "
        "No markdown, no prose. "
        "Return a JSON object with exactly one key: items. "
        "items must be an array of objects with exactly keys: section, title, url. "
        "Known music sections: "
        f"{music_csv}. "
        "Known books sections: "
        f"{books_csv}. "
        "If section is missing for music, choose the closest known music section; otherwise use Unsorted. "
        "For books, section may be blank because it can be inferred from Amazon taxonomy downstream. "
        "For books, title may be blank because it can be inferred from Amazon metadata downstream. "
        "If URL is missing, set url to an empty string."
    )
    user_prompt = (
        "Extract music and/or books from this request and return strict JSON:\n\n"
        f"{request_text}"
    )
    return [("system", system_prompt), ("user", user_prompt)]


def _safe_extract_json_dict(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned empty content.")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model output.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def extract_items(state: AgentState) -> AgentState:
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    llm_kwargs: dict[str, Any] = {
        "model": state["model_name"],
        "temperature": 0,
    }
    if openai_base_url:
        llm_kwargs["base_url"] = openai_base_url
    if openai_api_key:
        llm_kwargs["api_key"] = openai_api_key
    elif openai_base_url:
        # Local OpenAI-compatible endpoints (e.g., Ollama) usually accept any non-empty key.
        llm_kwargs["api_key"] = "ollama"

    llm = ChatOpenAI(**llm_kwargs)

    music_sections = SECTION_CATALOG_TOOL.list_sections()
    book_sections = BOOK_SECTION_CATALOG_TOOL.list_sections()
    messages = _make_extraction_messages(state["request_text"], music_sections, book_sections)

    next_state: AgentState = dict(state)
    next_state["extract_error"] = ""
    next_state["token_usage"] = {}

    try:
        structured_llm = llm.with_structured_output(
            ExtractionPayload, method="json_schema", strict=True, include_raw=True
        )
        response = structured_llm.invoke(messages)
        payload = response.get("parsed")
        raw_response = response.get("raw")

        usage = {}
        if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
            usage = {
                "input_tokens": int(raw_response.usage_metadata.get("input_tokens", 0)),
                "output_tokens": int(raw_response.usage_metadata.get("output_tokens", 0)),
                "total_tokens": int(raw_response.usage_metadata.get("total_tokens", 0)),
            }
        elif hasattr(raw_response, "response_metadata"):
            token_usage = (raw_response.response_metadata or {}).get("token_usage", {})
            usage = {
                "input_tokens": int(token_usage.get("prompt_tokens", 0)),
                "output_tokens": int(token_usage.get("completion_tokens", 0)),
                "total_tokens": int(token_usage.get("total_tokens", 0)),
            }
        next_state["token_usage"] = usage

        items: list[dict[str, str]] = []
        for item in payload.items if payload else []:
            items.append(
                {
                    "section": str(item.section),
                    "title": str(item.title),
                    "url": str(item.url),
                }
            )

        next_state["extracted_items"] = items
        return next_state
    except Exception as structured_exc:
        # Some local models do not support strict JSON schema. Fall back to plain invoke + JSON parse.
        try:
            raw_response = llm.invoke(messages)
            usage = {}
            if hasattr(raw_response, "usage_metadata") and raw_response.usage_metadata:
                usage = {
                    "input_tokens": int(raw_response.usage_metadata.get("input_tokens", 0)),
                    "output_tokens": int(raw_response.usage_metadata.get("output_tokens", 0)),
                    "total_tokens": int(raw_response.usage_metadata.get("total_tokens", 0)),
                }
            elif hasattr(raw_response, "response_metadata"):
                token_usage = (raw_response.response_metadata or {}).get("token_usage", {})
                usage = {
                    "input_tokens": int(token_usage.get("prompt_tokens", 0)),
                    "output_tokens": int(token_usage.get("completion_tokens", 0)),
                    "total_tokens": int(token_usage.get("total_tokens", 0)),
                }
            next_state["token_usage"] = usage

            content = raw_response.content if hasattr(raw_response, "content") else ""
            if isinstance(content, list):
                content = "".join(
                    str(chunk.get("text", "")) if isinstance(chunk, dict) else str(chunk) for chunk in content
                )
            parsed_payload = _safe_extract_json_dict(str(content))
            payload = ExtractionPayload.model_validate(parsed_payload)

            items: list[dict[str, str]] = []
            for item in payload.items if payload else []:
                items.append(
                    {
                        "section": str(item.section),
                        "title": str(item.title),
                        "url": str(item.url),
                    }
                )
            next_state["extracted_items"] = items
            return next_state
        except Exception as fallback_exc:
            next_state["extracted_items"] = []
            next_state["extract_error"] = (
                f"Extraction failed: structured={structured_exc}; fallback={fallback_exc}"
            )
            return next_state


def validate_items(state: AgentState) -> AgentState:
    valid_items: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []

    if state.get("extract_error"):
        invalid_items.append({"item": None, "reason": state["extract_error"]})

    items = state.get("extracted_items", [])
    if not items:
        invalid_items.append({"item": None, "reason": "No items extracted."})

    for idx, item in enumerate(items):
        section = str(item.get("section", "")).strip()
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()

        if not url:
            invalid_items.append({"item": item, "reason": f"Item {idx + 1}: url is required."})
            continue

        try:
            canonical_url = SPOTIFY_URL_NORMALIZER_TOOL.normalize(url)
            if not section:
                invalid_items.append({"item": item, "reason": f"Item {idx + 1}: section is required for music URLs."})
                continue
            parsed = music_ops.parse_spotify_url(canonical_url)
            if not parsed:
                raise ValueError("Could not parse Spotify URL.")
            kind, item_id, _ = parsed
            final_title = title or music_ops.fetch_spotify_title(kind, item_id) or music_ops.default_title(kind, item_id)
            valid_items.append({"item_type": "music", "section": section, "title": final_title, "url": canonical_url})
            continue
        except ValueError as exc:
            spotify_error = str(exc)

        try:
            canonical_url = AMAZON_URL_NORMALIZER_TOOL.normalize(url)
            final_title = title or books_ops.fetch_book_title(canonical_url) or books_ops.default_title(canonical_url)
            valid_items.append({"item_type": "book", "section": section, "title": final_title, "url": canonical_url})
        except ValueError as exc:
            invalid_items.append(
                {
                    "item": item,
                    "reason": f"Item {idx + 1}: invalid URL for supported content (spotify/amazon). "
                    f"Spotify error: {spotify_error}. Amazon error: {exc}",
                }
            )

    next_state: AgentState = dict(state)
    next_state["valid_items"] = valid_items
    next_state["invalid_items"] = invalid_items
    return next_state


def apply_upserts(state: AgentState) -> AgentState:
    results: list[dict[str, Any]] = []
    for item in state.get("valid_items", []):
        if item.get("item_type") == "book":
            result = BOOK_ADDER_TOOL.add_book(
                section=item["section"],
                title=item["title"],
                book_url=item["url"],
                dry_run=bool(state.get("dry_run", False)),
            )
        else:
            result = SONG_ADDER_TOOL.add_song(
                section=item["section"],
                title=item["title"],
                embed_url=item["url"],
                dry_run=bool(state.get("dry_run", False)),
            )
        results.append(result)

    next_state: AgentState = dict(state)
    next_state["results"] = results
    return next_state


def render(state: AgentState) -> AgentState:
    music_render_result = SITE_RENDERER_TOOL.render(dry_run=bool(state.get("dry_run", False)))
    books_render_result = BOOK_SITE_RENDERER_TOOL.render(dry_run=bool(state.get("dry_run", False)))
    render_result = {
        "status": "ok"
        if music_render_result.get("status") == "ok" and books_render_result.get("status") == "ok"
        else "error",
        "changed": bool(music_render_result.get("changed")) or bool(books_render_result.get("changed")),
        "reason": "",
        "music": music_render_result,
        "books": books_render_result,
    }
    next_state: AgentState = dict(state)
    next_state["render_result"] = render_result
    return next_state


def summarize(state: AgentState) -> AgentState:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"added": 0, "skipped": 0, "errors": 0})

    for result in state.get("results", []):
        section = str(result.get("section", "Unsorted"))
        status = str(result.get("status", "error"))
        if status == "added":
            grouped[section]["added"] += 1
        elif status == "skipped":
            grouped[section]["skipped"] += 1
        else:
            grouped[section]["errors"] += 1

    for section in sorted(grouped.keys(), key=str.casefold):
        counts = grouped[section]
        print(f"{section}: added {counts['added']}, skipped {counts['skipped']}, errors {counts['errors']}")

    print("Invalid items:")
    if state.get("invalid_items"):
        for item in state["invalid_items"]:
            print(json.dumps(item, ensure_ascii=True))
    else:
        print("[]")

    rr = state.get("render_result", {})
    status = rr.get("status", "error")
    changed = rr.get("changed", False)
    reason = rr.get("reason", "")
    print(f"Render: status={status}, changed={changed}, reason={reason}")
    if isinstance(rr.get("music"), dict):
        mr = rr["music"]
        print(f"Render music: status={mr.get('status')}, changed={mr.get('changed')}, reason={mr.get('reason', '')}")
    if isinstance(rr.get("books"), dict):
        br = rr["books"]
        print(f"Render books: status={br.get('status')}, changed={br.get('changed')}, reason={br.get('reason', '')}")

    token_usage = state.get("token_usage", {})
    if token_usage:
        print(
            "Token usage: "
            f"input={token_usage.get('input_tokens', 0)}, "
            f"output={token_usage.get('output_tokens', 0)}, "
            f"total={token_usage.get('total_tokens', 0)}"
        )

    return state


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("extract_items", extract_items)
    graph.add_node("validate_items", validate_items)
    graph.add_node("apply_upserts", apply_upserts)
    graph.add_node("render", render)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "extract_items")
    graph.add_edge("extract_items", "validate_items")
    graph.add_edge("validate_items", "apply_upserts")
    graph.add_edge("apply_upserts", "render")
    graph.add_edge("render", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def _read_request_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Either --text or --file is required.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph music ingestion agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Natural language request text")
    group.add_argument("--file", help="Path to file containing request text")
    parser.add_argument("--dry-run", action="store_true", help="Validate and simulate without writing files")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="Model name (OpenAI model or local OpenAI-compatible model ID)",
    )
    args = parser.parse_args()

    request_text = _read_request_text(args)

    app = build_graph()
    initial_state: AgentState = {
        "request_text": request_text,
        "extracted_items": [],
        "valid_items": [],
        "invalid_items": [],
        "results": [],
        "render_result": {},
        "model_name": args.model,
        "dry_run": bool(args.dry_run),
        "extract_error": "",
        "token_usage": {},
    }

    app.invoke(initial_state)


if __name__ == "__main__":
    main()
