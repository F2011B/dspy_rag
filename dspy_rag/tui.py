from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.widgets import TextArea, Frame

from .rag import AnswerResult, RAGAppState, Source


@dataclass
class ChatMessage:
    role: str
    text: str


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return "No sources."
    lines = []
    for source in sources:
        line_range = ""
        if source.start_line and source.end_line:
            line_range = f" (lines {source.start_line}-{source.end_line})"
        lines.append(f"[{source.number}] {source.rel_path}{line_range}")
        lines.append(f"    {source.snippet}")
        lines.append("")
    return "\n".join(lines).strip()


def format_chat(messages: list[ChatMessage]) -> str:
    lines = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant" if msg.role == "assistant" else "System"
        lines.append(f"{prefix}:\n{msg.text}\n")
    return "\n".join(lines).strip()


class StatusSpinner:
    def __init__(self, set_status: Callable[[str], None]):
        self.set_status = set_status
        self._task: asyncio.Task | None = None
        self._running = False
        self._label = ""

    async def _spin(self) -> None:
        symbols = "|/-\\"
        idx = 0
        while self._running:
            self.set_status(f"{self._label} {symbols[idx % len(symbols)]}")
            idx += 1
            await asyncio.sleep(0.12)

    def start(self, label: str) -> None:
        self.stop("")
        self._label = label
        self._running = True
        self._task = asyncio.create_task(self._spin())

    def stop(self, message: str = "") -> None:
        if self._running:
            self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.set_status(message)


class FolderRAGTUI:
    def __init__(self, app_state: RAGAppState):
        self.app_state = app_state
        self.messages: list[ChatMessage] = []
        self.last_sources: list[Source] = []

        self.chat_area = TextArea(
            text="",
            focusable=False,
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
        )
        self.sources_area = TextArea(
            text="",
            focusable=False,
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
        )
        self.input_area = TextArea(
            text="",
            multiline=True,
            wrap_lines=True,
            height=Dimension(min=3, max=8),
        )

        self.status_control = FormattedTextControl(text=self.default_status())
        self.status_bar = Window(height=1, content=self.status_control)

        self.spinner = StatusSpinner(self.set_status)

        body = VSplit(
            [
                Frame(self.chat_area, title="Chat", width=Dimension(weight=3, min=40)),
                Frame(self.sources_area, title="Sources", width=Dimension(weight=1, min=30)),
            ],
            padding=1,
        )
        layout = HSplit(
            [
                body,
                Frame(self.input_area, title="Input"),
                self.status_bar,
            ]
        )
        self.kb = self._build_keybindings()
        self.app = Application(
            layout=Layout(layout, focused_element=self.input_area),
            key_bindings=self.kb,
            full_screen=True,
        )

    def default_status(self) -> str:
        return "F1 help | F2 send | Ctrl+C exit"

    def set_status(self, text: str) -> None:
        self.status_control.text = text
        self.app.invalidate()

    def append_message(self, role: str, text: str) -> None:
        self.messages.append(ChatMessage(role=role, text=text.strip()))
        self.chat_area.text = format_chat(self.messages)
        self.chat_area.buffer.cursor_position = len(self.chat_area.text)

    def update_sources(self, sources: list[Source]) -> None:
        self.last_sources = sources
        self.sources_area.text = format_sources(sources)
        self.sources_area.buffer.cursor_position = len(self.sources_area.text)

    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("f2")
        def _send(event) -> None:
            event.app.create_background_task(self.handle_send())

        @kb.add("f1")
        def _help(event) -> None:
            self.show_help()

        @kb.add("c-c")
        def _exit(event) -> None:
            event.app.exit()

        return kb

    def show_help(self) -> None:
        help_text = (
            "Keybindings:\n"
            "- F1: help\n"
            "- F2: send\n"
            "- Ctrl+C: exit\n\n"
            "Commands:\n"
            "/help, /exit, /reindex, /reload, /clear, /stats, /sources\n\n"
            "Tips:\n"
            "- Enter inserts a new line.\n"
        )
        self.append_message("system", help_text)

    async def handle_send(self) -> None:
        text = self.input_area.text.strip()
        if not text:
            return
        self.input_area.text = ""
        if text.startswith("/"):
            await self.handle_command(text)
            return
        self.append_message("user", text)
        await self.run_query(text)

    async def handle_command(self, text: str) -> None:
        command = text.strip().lower()
        if command == "/exit":
            self.app.exit()
            return
        if command == "/help":
            self.show_help()
            return
        if command == "/clear":
            self.messages.clear()
            self.chat_area.text = ""
            return
        if command == "/sources":
            self.update_sources(self.last_sources)
            self.append_message("system", "Sources refreshed.")
            return
        if command == "/stats":
            stats = self.app_state.stats_payload()
            stats_text = "\n".join(
                [
                    f"files: {stats['files']}",
                    f"chunks: {stats['chunks']}",
                    f"bytes: {stats['bytes']}",
                    f"root_dir: {stats['root_dir']}",
                    f"index_dir: {stats['index_dir']}",
                    f"chunk_size: {stats['chunk_size']}",
                    f"chunk_overlap: {stats['chunk_overlap']}",
                    f"topk: {stats['topk']}",
                    f"model: {stats['model']}",
                    f"embed_model: {stats['embed_model']}",
                ]
            )
            self.append_message("system", stats_text)
            return
        if command == "/reload":
            self.spinner.start("Reloading index")
            try:
                loaded = await asyncio.to_thread(self.app_state.reload)
            finally:
                self.spinner.stop(self.default_status())
            if loaded:
                self.append_message("system", "Index reloaded.")
            else:
                self.append_message("system", "No index found to reload.")
            return
        if command == "/reindex":
            self.spinner.start("Reindexing")
            try:
                await asyncio.to_thread(self.app_state.reindex)
            except Exception as exc:
                self.append_message("system", f"Reindex failed: {exc}")
            else:
                self.append_message("system", "Reindex completed.")
            finally:
                self.spinner.stop(self.default_status())
            return

        self.append_message("system", f"Unknown command: {text}")

    async def run_query(self, question: str) -> None:
        max_attempts = 4
        last_result: AnswerResult | None = None

        for attempt in range(1, max_attempts + 1):
            self.spinner.start(f"Thinking (attempt {attempt}/{max_attempts})")
            try:
                result: AnswerResult = await asyncio.to_thread(self.app_state.answer, question)
            except Exception as exc:
                self.append_message("system", f"Error: {exc}")
                self.spinner.stop(self.default_status())
                return

            if result.answer.strip():
                self.spinner.stop(self.default_status())
                self.append_message("assistant", result.answer)
                self.update_sources(result.sources)
                return

            last_result = result
            if attempt < max_attempts:
                self.spinner.stop(f"No answer received, retrying {attempt + 1}/{max_attempts}")
                await asyncio.sleep(0.15)

        self.spinner.stop("No answer after retries")
        self.append_message(
            "system",
            "Model returned no answer after 4 attempts. Please try again or rephrase.",
        )
        if last_result and last_result.sources:
            self.update_sources(last_result.sources)

    def run(self) -> None:
        self.app.run()
