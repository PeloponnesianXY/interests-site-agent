from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import music_ops


class ExtractedItem(BaseModel):
    section: str = Field(default="")
    title: str = Field(default="")
    url: str = Field(default="")


class ExtractionPayload(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    request_text: str
    extracted_items: list[dict[str, str]]
    valid_items: list[dict[str, str]]
    invalid_items: list[dict[str, Any]]
    results: list[dict[str, Any]]
    render_result: dict[str, Any]
    model_name: str
    dry_run: bool
    extract_error: str


def _make_extraction_messages(request_text: str, section_names: list[str]) -> list[tuple[str, str]]:
    section_csv = ", ".join(section_names) if section_names else "(none)"
    system_prompt = (
        "You are an extraction engine. Output STRICT JSON only. "
        "No markdown, no prose. "
        "Return a JSON object with exactly one key: items. "
        "items must be an array of objects with exactly keys: section, title, url. "
        "Known sections: "
        f"{section_csv}. "
        "If section is missing, choose the closest known section; otherwise use Unsorted. "
        "If URL is missing, set url to an empty string."
    )
    user_prompt = (
        "Extract songs from this request and return strict JSON:\n\n"
        f"{request_text}"
    )
    return [("system", system_prompt), ("user", user_prompt)]


def extract_items(state: AgentState) -> AgentState:
    llm = ChatOpenAI(
        model=state["model_name"],
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    section_names = music_ops.get_section_names()
    messages = _make_extraction_messages(state["request_text"], section_names)

    next_state: AgentState = dict(state)
    next_state["extract_error"] = ""

    try:
        structured_llm = llm.with_structured_output(ExtractionPayload, method="json_schema", strict=True)
        payload = structured_llm.invoke(messages)

        items: list[dict[str, str]] = []
        for item in payload.items:
            items.append(
                {
                    "section": str(item.section),
                    "title": str(item.title),
                    "url": str(item.url),
                }
            )

        next_state["extracted_items"] = items
        return next_state
    except Exception as exc:
        next_state["extracted_items"] = []
        next_state["extract_error"] = f"Extraction failed: {exc}"
        return next_state


def validate_items(state: AgentState) -> AgentState:
    valid_items: list[dict[str, str]] = []
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

        if not section:
            invalid_items.append({"item": item, "reason": f"Item {idx + 1}: section is required."})
            continue

        if not title:
            invalid_items.append({"item": item, "reason": f"Item {idx + 1}: title is required."})
            continue

        if not url:
            invalid_items.append({"item": item, "reason": f"Item {idx + 1}: url is required."})
            continue

        ok, detail = music_ops.validate_spotify_embed_url(url)
        if not ok:
            invalid_items.append({"item": item, "reason": f"Item {idx + 1}: {detail}"})
            continue

        valid_items.append({"section": section, "title": title, "url": detail})

    next_state: AgentState = dict(state)
    next_state["valid_items"] = valid_items
    next_state["invalid_items"] = invalid_items
    return next_state


def apply_upserts(state: AgentState) -> AgentState:
    results: list[dict[str, Any]] = []
    for item in state.get("valid_items", []):
        result = music_ops.upsert_song(
            section=item["section"],
            title=item["title"],
            url=item["url"],
            dry_run=bool(state.get("dry_run", False)),
        )
        results.append(result)

    next_state: AgentState = dict(state)
    next_state["results"] = results
    return next_state


def render(state: AgentState) -> AgentState:
    render_result = music_ops.render_site(dry_run=bool(state.get("dry_run", False)))
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
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="Model name")
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
    }

    app.invoke(initial_state)


if __name__ == "__main__":
    main()
