#!/usr/bin/env python3
"""Slack bot that converts document attachments (PDF, DOCX, TXT, HTML) to Markdown when a user reacts with an emoji."""

import os
import logging
import tempfile

import requests
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from convert import convert_file_bytes, split_markdown

SUPPORTED_TYPES = {"pdf", "docx", "text", "html"}

# Map file extensions to our internal type when Slack's filetype doesn't match
EXT_TO_TYPE = {".pdf": "pdf", ".docx": "docx", ".txt": "text", ".html": "html", ".htm": "html"}

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
CONVERT_EMOJI = os.environ.get("CONVERT_EMOJI", "page_facing_up")

app = App(token=SLACK_BOT_TOKEN)


def download_file(url: str) -> bytes:
    """Download a Slack-hosted file using the bot token for auth."""
    resp = requests.get(url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"})
    resp.raise_for_status()
    return resp.content


@app.event("reaction_added")
def handle_reaction(event, client, logger):
    logger.info("Reaction received: '%s' (looking for: '%s')", event["reaction"], CONVERT_EMOJI)
    if event["reaction"] != CONVERT_EMOJI:
        return

    channel = event["item"]["channel"]
    message_ts = event["item"]["ts"]

    # Fetch the message that was reacted to
    result = client.conversations_history(
        channel=channel, latest=message_ts, inclusive=True, limit=1
    )
    if not result["messages"]:
        # Might be a threaded reply — try conversations.replies
        result = client.conversations_replies(
            channel=channel, ts=message_ts, inclusive=True, limit=1
        )
    if not result["messages"]:
        logger.warning("Could not find message for ts=%s", message_ts)
        return

    message = result["messages"][0]
    files = message.get("files", [])

    # Resolve each file's type: prefer Slack's filetype, fall back to extension
    def resolve_type(f):
        ft = f.get("filetype", "")
        if ft in SUPPORTED_TYPES:
            return ft
        name = f.get("name", "")
        ext = os.path.splitext(name)[1].lower()
        return EXT_TO_TYPE.get(ext)

    supported = [(f, resolve_type(f)) for f in files if resolve_type(f)]

    if not supported:
        logger.info("Reaction on message ts=%s but no supported files found", message_ts)
        return

    for doc_file, filetype in supported:
        name = doc_file.get("name", "document")
        md_filename = name.rsplit(".", 1)[0] + ".md"

        try:
            file_bytes = download_file(doc_file["url_private_download"])
            logger.info("Downloaded %s (%s): %d bytes", name, filetype, len(file_bytes))
            md_text, report = convert_file_bytes(file_bytes, filename=name, filetype=filetype)
            logger.info("Converted %s: %d chars of markdown", name, len(md_text))
        except Exception:
            logger.exception("Failed to convert %s", name)
            client.chat_postMessage(
                channel=channel,
                thread_ts=message_ts,
                text=f"Sorry, I couldn't convert `{name}` to Markdown.",
            )
            continue

        if not md_text.strip():
            logger.warning("Conversion of %s produced empty markdown (possibly a scanned/image PDF)", name)
            ocr_msg = ""
            if report.ocr_needed:
                ocr_msg = (
                    f" OCR was attempted on {len(report.ocr_pages)} page(s)"
                    " but extraction still produced no text."
                )
            client.chat_postMessage(
                channel=channel,
                thread_ts=message_ts,
                text=(
                    f"The conversion of `{name}` produced no text.{ocr_msg}"
                    " This may be a scanned document that OCR couldn't process."
                ),
            )
            continue

        # Split output if MAX_OUTPUT_CHARS is configured
        max_chars_str = os.environ.get("MAX_OUTPUT_CHARS", "")
        max_chars = int(max_chars_str) if max_chars_str.strip().isdigit() else 0
        base_name = name.rsplit(".", 1)[0]

        if max_chars and len(md_text) > max_chars:
            parts = split_markdown(md_text, max_chars)
            total = len(parts)
            for idx, part in enumerate(parts, 1):
                part_filename = f"{base_name}_part{idx}of{total}.md"
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
                    tmp.write(part)
                    tmp_path = tmp.name
                try:
                    comment = (
                        f"Part {idx}/{total} of `{name}` ({len(part):,} chars):"
                        if idx == 1
                        else f"Part {idx}/{total}:"
                    )
                    client.files_upload_v2(
                        channel=channel,
                        thread_ts=message_ts,
                        file=tmp_path,
                        filename=part_filename,
                        initial_comment=comment,
                    )
                finally:
                    os.unlink(tmp_path)
        else:
            # Upload the .md file via a temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
                tmp.write(md_text)
                tmp_path = tmp.name
            try:
                client.files_upload_v2(
                    channel=channel,
                    thread_ts=message_ts,
                    file=tmp_path,
                    filename=md_filename,
                    initial_comment=f"Here's the Markdown conversion of `{name}`:",
                )
            finally:
                os.unlink(tmp_path)

        # Upload the conversion report
        report_filename = name.rsplit(".", 1)[0] + "_report.md"
        report_md = report.to_markdown()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp_report:
            tmp_report.write(report_md)
            tmp_report_path = tmp_report.name
        try:
            client.files_upload_v2(
                channel=channel,
                thread_ts=message_ts,
                file=tmp_report_path,
                filename=report_filename,
                initial_comment=f"Conversion report for `{name}`:",
            )
        finally:
            os.unlink(tmp_report_path)

        # Post a brief inline summary
        summary_parts = [f"Confidence: *{report.confidence}*"]
        summary_parts.append(
            f"Words: {report.source_words:,} source -> "
            f"{report.output_words:,} output ({report.retention_pct}%)"
        )
        if report.ocr_needed:
            summary_parts.append(
                f"OCR: {len(report.ocr_pages)}/{report.total_pages} pages"
            )
        if report.missing_toc_entries:
            summary_parts.append(
                f"Missing sections: {len(report.missing_toc_entries)}"
            )
        summary = " | ".join(summary_parts)
        client.chat_postMessage(
            channel=channel,
            thread_ts=message_ts,
            text=summary,
        )

        # Preview snippet removed — the .md file and report are sufficient


# Acknowledge other events so Bolt doesn't log warnings for each one
@app.event("message")
def handle_message(event, logger):
    pass


@app.event("file_shared")
def handle_file_shared(event, logger):
    pass


@app.event("file_created")
def handle_file_created(event, logger):
    pass


@app.event("file_public")
def handle_file_public(event, logger):
    pass


@app.event("file_deleted")
def handle_file_deleted(event, logger):
    pass


@app.event("app_mention")
def handle_app_mention(event, logger):
    pass


if __name__ == "__main__":
    logger.info("Starting Slack PDF-to-Markdown bot (emoji: :%s:)", CONVERT_EMOJI)
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
