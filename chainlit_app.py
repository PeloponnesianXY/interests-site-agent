from __future__ import annotations

import asyncio
import os
import threading

import chainlit as cl

from agent_music_langgraph import AgentState, build_graph


APP = build_graph()
INGESTION_LOCK = threading.Lock()


def _default_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _initial_state(request_text: str) -> AgentState:
    return {
        "request_text": request_text,
        "extracted_items": [],
        "valid_items": [],
        "invalid_items": [],
        "results": [],
        "render_result": {},
        "model_name": _default_model_name(),
        "dry_run": False,
        "extract_error": "",
        "token_usage": {},
    }


def _run_agent(request_text: str) -> AgentState:
    initial_state = _initial_state(request_text)
    with INGESTION_LOCK:
        return APP.invoke(initial_state)


def _format_result(state: AgentState) -> str:
    lines: list[str] = []

    added = [result for result in state.get("results", []) if result.get("status") == "added"]
    skipped = [result for result in state.get("results", []) if result.get("status") == "skipped"]
    errors = [result for result in state.get("results", []) if result.get("status") == "error"]
    invalid_items = state.get("invalid_items", [])
    render_result = state.get("render_result", {})

    lines.append(
        f"Processed {len(state.get('extracted_items', []))} extracted item(s): "
        f"{len(added)} added, {len(skipped)} skipped, {len(errors) + len(invalid_items)} error(s)."
    )

    if added:
        lines.append("")
        lines.append("Added:")
        for result in added:
            lines.append(
                f"- [{result.get('section', 'Unsorted')}] {result.get('title', 'Untitled')} ({result.get('url', '')})"
            )

    if skipped:
        lines.append("")
        lines.append("Skipped:")
        for result in skipped:
            lines.append(
                f"- [{result.get('section', 'Unsorted')}] {result.get('title', 'Untitled')}: {result.get('reason', '')}"
            )

    if errors:
        lines.append("")
        lines.append("Write errors:")
        for result in errors:
            lines.append(
                f"- [{result.get('section', 'Unsorted')}] {result.get('title', 'Untitled')}: {result.get('reason', '')}"
            )

    if invalid_items:
        lines.append("")
        lines.append("Invalid items:")
        for item in invalid_items:
            reason = str(item.get("reason", "Unknown error"))
            raw_item = item.get("item")
            if isinstance(raw_item, dict) and raw_item.get("url"):
                lines.append(f"- {raw_item.get('url')}: {reason}")
            else:
                lines.append(f"- {reason}")

    if render_result:
        lines.append("")
        lines.append(
            "Render status: "
            f"{render_result.get('status', 'error')} "
            f"(changed={bool(render_result.get('changed', False))})"
        )

    token_usage = state.get("token_usage", {})
    if token_usage:
        lines.append(
            "Token usage: "
            f"{token_usage.get('input_tokens', 0)} in / "
            f"{token_usage.get('output_tokens', 0)} out / "
            f"{token_usage.get('total_tokens', 0)} total"
        )

    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "Drop or paste URLs directly into the chat.\n"
            "You can send 1 or several mixed Amazon and Spotify links in one message.\n"
            "For Spotify links, include the genre or section you want, for example: "
            "'Add these to Modern Indie: <spotify links>'. Amazon links go to books automatically."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    request_text = (message.content or "").strip()
    if not request_text:
        await cl.Message(
            content="No URL text was provided. Paste or drop the URLs directly into the chat message."
        ).send()
        return

    progress = cl.Message(content="Processing links and updating the site...")
    await progress.send()

    try:
        state = await asyncio.to_thread(_run_agent, request_text)
    except UnicodeDecodeError as exc:
        progress.content = f"Could not read one of the attached files as UTF-8: {exc}"
        await progress.update()
        return
    except Exception as exc:
        progress.content = f"Agent run failed: {exc}"
        await progress.update()
        return

    progress.content = _format_result(state)
    progress.elements = [
        cl.File(name="music.json", path="music.json", display="inline"),
        cl.File(name="books.json", path="books.json", display="inline"),
        cl.File(name="index.html", path="index.html", display="inline"),
        cl.File(name="books.html", path="books.html", display="inline"),
    ]
    await progress.update()
