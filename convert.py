#!/usr/bin/env python3
"""Convert documents (PDF, DOCX, TXT) to Markdown — used by both the CLI and Slack bot."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import mammoth
import ocrmypdf
import pymupdf4llm
from anthropic import Anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversion report
# ---------------------------------------------------------------------------

@dataclass
class ConversionReport:
    filename: str
    filetype: str
    # OCR
    ocr_needed: bool = False
    ocr_pages: list = field(default_factory=list)
    total_pages: int = 0
    # Content stats
    source_words: int = 0
    output_words: int = 0
    retention_pct: float = 0.0
    per_stage_words: dict = field(default_factory=dict)
    # Structure
    toc_entries_total: int = 0
    missing_toc_entries: list = field(default_factory=list)
    # LLM chunk issues
    chunk_issues: list = field(default_factory=list)
    chunks_fell_back: int = 0
    # Per-chunk detail: list of {"chunk_num", "status", "reason", "start_line", "end_line"}
    chunk_details: list = field(default_factory=list)
    # Heading suggestions (matched TOC entries found in body)
    heading_suggestions: list = field(default_factory=list)
    # Page artifact stats
    page_artifact_stats: dict = field(default_factory=dict)
    # Front/back matter boundaries
    front_matter_end_line: int | None = None
    front_matter_summary: str = ""
    back_matter_start_line: int | None = None
    back_matter_summary: str = ""
    # Overall
    confidence: str = "unknown"

    def to_markdown(self) -> str:
        lines = [f"# Conversion Report: {self.filename}", ""]

        # OCR
        lines.append("## OCR Status")
        if self.filetype != "pdf":
            lines.append("- N/A (not a PDF)")
        elif self.ocr_needed:
            lines.append(
                f"- OCR was applied to {len(self.ocr_pages)}/{self.total_pages} pages"
            )
            if len(self.ocr_pages) <= 20:
                pages_str = ", ".join(str(p + 1) for p in self.ocr_pages)
                lines.append(f"- Pages OCR'd: {pages_str}")
        else:
            lines.append("- No OCR needed (all pages had extractable text)")
        lines.append("")

        # Content coverage
        lines.append("## Content Coverage")
        lines.append(f"- Source words: {self.source_words:,}")
        lines.append(f"- Output words: {self.output_words:,}")
        lines.append(f"- Retention: {self.retention_pct}%")
        if self.per_stage_words:
            lines.append("- Per-stage word counts:")
            for stage, count in self.per_stage_words.items():
                lines.append(f"  - {stage}: {count:,}")
        if self.chunks_fell_back > 0:
            lines.append(
                f"- **{self.chunks_fell_back} chunk(s) fell back to original** "
                "(LLM dropped too much content)"
            )
        # Page artifact stats
        if self.page_artifact_stats:
            pg = self.page_artifact_stats.get("page_numbers_commented", 0)
            hd = self.page_artifact_stats.get("running_headers_commented", 0)
            if pg or hd:
                parts = []
                if pg:
                    parts.append(f"{pg} page numbers")
                if hd:
                    parts.append(f"{hd} running headers")
                lines.append(
                    f"- Converted to HTML comments: {', '.join(parts)} "
                    "(preserved as metadata, hidden from text processing)"
                )
        # Discrepancy explanation
        if self.source_words and self.output_words:
            diff = self.source_words - self.output_words
            if diff > 0:
                lines.append("")
                lines.append(f"**Word count difference ({diff:,} words) likely due to:**")
                reasons = []
                pg = self.page_artifact_stats.get("page_numbers_commented", 0) if self.page_artifact_stats else 0
                hd = self.page_artifact_stats.get("running_headers_commented", 0) if self.page_artifact_stats else 0
                if pg or hd:
                    reasons.append(f"Page numbers and running headers commented out (~{pg + hd} lines)")
                if self.chunks_fell_back > 0:
                    reasons.append(f"LLM reflowed/cleaned {len(self.chunk_details) - self.chunks_fell_back} chunks (minor word count changes from fixing broken words, removing artifacts)")
                if self.per_stage_words:
                    raw = self.per_stage_words.get("raw", 0)
                    after_regex = self.per_stage_words.get("after_regex", 0)
                    if raw > after_regex:
                        reasons.append(f"Regex cleanup removed {raw - after_regex:,} words (control chars, artifacts)")
                if not reasons:
                    reasons.append("LLM cleanup normalized whitespace and removed conversion artifacts")
                for r in reasons:
                    lines.append(f"- {r}")
        lines.append("")

        # Commented headers detail
        commented = self.page_artifact_stats.get("headers_commented", []) if self.page_artifact_stats else []
        excluded = self.page_artifact_stats.get("headers_excluded", []) if self.page_artifact_stats else []
        if commented or excluded:
            lines.append("## Commented Running Headers")
            lines.append(
                "The following repeated lines were converted to "
                "`<!-- header: ... -->` HTML comments. Review the list "
                "to verify no meaningful content was caught."
            )
            lines.append("")
            if commented:
                lines.append("**Commented out:**")
                for text, count in commented:
                    lines.append(f"- `{text}` ({count}×)")
                lines.append("")
            if excluded:
                lines.append(
                    "**Kept as content** (repeated 3+ times but looked "
                    "like dialogue or sentences — NOT commented out):"
                )
                for text, count in excluded:
                    lines.append(f"- `{text}` ({count}×)")
                lines.append("")
            lines.append(
                "*If a header was wrongly commented out, search the .md "
                "for `<!-- header: ... -->` and remove the comment wrapper. "
                "If content was wrongly kept, wrap it in "
                "`<!-- header: ... -->` manually.*"
            )
            lines.append("")

        # Structure
        if self.toc_entries_total > 0:
            lines.append("## Structure Check")
            lines.append(f"- TOC entries in source: {self.toc_entries_total}")
            if self.missing_toc_entries:
                lines.append(
                    f"- **Missing in output ({len(self.missing_toc_entries)}):**"
                )
                for entry in self.missing_toc_entries:
                    lines.append(f"  - {entry}")
            else:
                lines.append("- All TOC entries found in output")
            lines.append("")

        # Heading suggestions
        if self.heading_suggestions:
            lines.append("## Heading Suggestions")
            lines.append(
                "The following lines in the output appear to match TOC entries "
                "but are not formatted as Markdown headings. Review the matches "
                "below — if they look correct, you can use the prompt at the "
                "bottom of this section to fix them automatically with an LLM."
            )
            lines.append("")
            for sug in self.heading_suggestions:
                lines.append(
                    f"- **Line {sug['line_num']}**: `{sug['line_text'][:80]}` "
                    f"→ TOC entry: *{sug['toc_title']}*"
                )
            lines.append("")
            lines.append("### How to apply")
            lines.append(
                "After reviewing the matches above, copy the prompt below "
                "and paste it into an LLM along with the .md file:"
            )
            lines.append("")
            lines.append("```")
            lines.append(
                "The following lines in the attached .md file should be "
                "promoted to Markdown headings. For each match, replace the "
                "existing line with a proper heading at the level indicated. "
                "Do not change anything else in the file."
            )
            lines.append("")
            for sug in self.heading_suggestions:
                lines.append(
                    f"- Line {sug['line_num']}: "
                    f"Change `{sug['line_text'][:80]}` to "
                    f"`## {sug['toc_title']}`"
                )
            lines.append("```")
            lines.append("")

        # Front/back matter
        if self.front_matter_end_line is not None or self.back_matter_start_line is not None:
            lines.append("## Document Boundaries")
            lines.append(
                "HTML comment markers have been inserted in the .md file to "
                "delineate front matter, primary text, and back matter. "
                "You can move or remove these markers as needed."
            )
            lines.append("")
            if self.front_matter_end_line is not None:
                lines.append(
                    f"- **Front matter ends** at line {self.front_matter_end_line}"
                )
                if self.front_matter_summary:
                    lines.append(f"  - Contains: {self.front_matter_summary}")
            else:
                lines.append("- No front matter detected")
            if self.back_matter_start_line is not None:
                lines.append(
                    f"- **Back matter starts** at line {self.back_matter_start_line}"
                )
                if self.back_matter_summary:
                    lines.append(f"  - Contains: {self.back_matter_summary}")
            else:
                lines.append("- No back matter detected")
            lines.append("")
            lines.append("**Marker reference:**")
            lines.append("- `<!-- FRONT_MATTER_END -->` — end of front matter")
            lines.append(
                "- `<!-- BACK_MATTER_START -->` — start of back matter"
            )
            lines.append(
                "- To search only the primary text, extract content between "
                "these two markers"
            )
            lines.append("")

        # Chunks that need manual attention
        failed_chunks = [
            d for d in self.chunk_details if d["status"] in ("failed", "skipped")
        ]
        if failed_chunks:
            lines.append("## Chunks Needing Manual Cleanup")
            lines.append(
                "The following sections were not cleaned by the LLM. "
                "The original extracted text is preserved, but may still "
                "contain OCR artifacts, broken words, or formatting issues."
            )
            lines.append("")
            for cd in failed_chunks:
                lines.append(
                    f"- **Chunk {cd['chunk_num']}** "
                    f"(lines {cd['start_line']}–{cd['end_line']}, "
                    f"{cd['words']:,} words): {cd['reason']}"
                )
            lines.append("")
            lines.append("### How to fix manually")
            lines.append(
                "1. Open the .md file and copy the lines listed above "
                "for the chunk you want to fix"
            )
            lines.append(
                "2. Paste the copied text into an LLM (e.g. Claude) "
                "along with the prompt below"
            )
            lines.append(
                "3. Replace the original lines in the .md file with "
                "the cleaned output"
            )
            lines.append("")
            lines.append("```")
            lines.append(
                "Clean up this text extracted from a scholarly/academic PDF. "
                "Fix garbled or broken words, remove stray characters and "
                "page numbers, and fix broken line breaks. "
                "CRITICAL: Do NOT delete, summarize, or skip any content. "
                "Every sentence must be preserved. Output only the "
                "cleaned text."
            )
            lines.append("```")
            lines.append("")

        # LLM issues (filter out raw API error strings — those are covered above)
        all_issues = [
            iss
            for chunk in self.chunk_issues
            for iss in chunk.get("issues", [])
            if not iss.startswith("LLM cleanup unavailable:")
        ]
        if all_issues:
            lines.append("## Issues Found During Cleanup")
            for issue in all_issues:
                lines.append(f"- {issue}")
            lines.append("")

        # Confidence
        lines.append("## Overall Confidence")
        label = {"high": "Good", "medium": "Fair", "low": "Poor"}.get(
            self.confidence, "Unknown"
        )
        lines.append(f"**{label}** ({self.confidence})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLEANUP_PROMPT = """\
You are a document cleanup assistant working in an academic/scholarly setting. \
The texts you process are source materials for university research and teaching — \
they may include philosophy, history, literature, religion, and other humanities \
subjects spanning all time periods and cultures. Your HIGHEST PRIORITY is \
preserving every piece of original content with full fidelity. The following \
markdown was extracted from a document and may contain conversion artifacts.

BEFORE CLEANING, assess what kind of text you are looking at. The text may contain \
any mix of the following — treat each section according to its type:

PROSE (essays, treatises, narrative, technical descriptions):
- Reflow hard-wrapped lines into continuous paragraphs
- A line that ends mid-sentence and continues on the next line is a hard wrap, not \
a meaningful break

POETRY / VERSE (epics, hymns, odes, songs, metrical text):
- PRESERVE every line break exactly as it appears — each line is a deliberate unit
- Do NOT reflow verse into prose paragraphs
- Preserve stanza breaks (blank lines between groups of lines)
- Signals: short lines of similar length, metrical rhythm, rhyme, refrains, \
line-initial capitalization

DIALOGUE / DRAMA (speeches, choruses, speaker tags):
- Preserve speaker labels (e.g. "CHORUS:", "SOCRATES:") on their own lines
- Keep each speech turn as a distinct block
- Preserve responsive patterns (e.g. liturgical call-and-response)

TABLES / LISTS / INDEX ENTRIES:
- Preserve columnar alignment and list structure

If a section is ambiguous, ERR ON THE SIDE OF PRESERVING LINE BREAKS — it is \
better to leave a hard wrap in prose than to destroy a line of verse.

CLEANUP TASKS:
1. Fix garbled or broken words (OCR/extraction errors)
2. Remove stray characters, repeated punctuation artifacts, and control characters
3. Fix broken line breaks (words split across lines with hyphens)
4. Remove running headers and footers (author names, book titles, page numbers \
that repeat at page boundaries) — but keep lines that are meaningful content
5. Preserve the original structure (headings, lists, paragraphs, tables)

CRITICAL RULES:
- NEVER delete, summarize, condense, or skip any substantive content
- NEVER merge or combine separate paragraphs or sections
- Every sentence in the input must appear in the output
- When in doubt about whether something is an artifact or real content, KEEP IT
- Your output should be roughly the same length as the input

ISSUE REPORTING:
On the very first line of your response, output a JSON object in this exact format:
ISSUES_JSON:{"issues": ["list of any problems you noticed"], "garbled_words_fixed": 0}
If you found no issues, use: ISSUES_JSON:{"issues": [], "garbled_words_fixed": 0}
Then a blank line, then the cleaned markdown. Do NOT wrap the markdown in code fences."""


# ---------------------------------------------------------------------------
# OCR pre-processing
# ---------------------------------------------------------------------------

_ocr_lock = threading.Lock()


def _ocr_pdf_if_needed(pdf_bytes: bytes) -> tuple[bytes, dict]:
    """Run ocrmypdf on the PDF if any pages lack a text layer.

    Returns (output_pdf_bytes, ocr_info_dict).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages_without_text = []
    for i, page in enumerate(doc):
        if not page.get_text().strip():
            pages_without_text.append(i)
    doc.close()

    ocr_info = {
        "needed": len(pages_without_text) > 0,
        "pages_ocrd": pages_without_text,
        "total_pages": total_pages,
    }

    if not pages_without_text:
        return pdf_bytes, ocr_info

    logger.info(
        "OCR needed: %d/%d pages lack text — running ocrmypdf",
        len(pages_without_text),
        total_pages,
    )
    input_buf = io.BytesIO(pdf_bytes)
    output_buf = io.BytesIO()

    with _ocr_lock:
        ocrmypdf.ocr(
            input_buf,
            output_buf,
            skip_text=True,
            language="eng",
            progress_bar=False,
            output_type="pdf",
        )

    return output_buf.getvalue(), ocr_info


# ---------------------------------------------------------------------------
# Structure-aware PDF extraction
# ---------------------------------------------------------------------------


def _extract_pdf_with_structure(pdf_path: str) -> tuple[str, list, dict]:
    """Extract markdown from a PDF with TOC and per-page word counts.

    Returns (markdown_text, toc_entries, source_stats).
    toc_entries: list of [level, title, page_num] from PyMuPDF.
    source_stats: {"per_page_words": [...], "total_words": int}.

    Uses pymupdf4llm for markdown extraction. If pymupdf4llm returns
    empty results (common with OCR-only PDFs), falls back to plain
    text extraction via fitz.get_text().
    """
    doc = fitz.open(pdf_path)
    toc_entries = doc.get_toc()

    source_stats = {"per_page_words": [], "total_words": 0}
    fallback_pages: list[str] = []
    for page in doc:
        text = page.get_text()
        words = text.split()
        source_stats["per_page_words"].append(len(words))
        source_stats["total_words"] += len(words)
        fallback_pages.append(text.strip())
    doc.close()

    # Primary extraction via pymupdf4llm
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    md_parts = []
    for chunk in chunks:
        page_md = chunk.get("text", "")
        if page_md.strip():
            md_parts.append(page_md)

    markdown_text = "\n\n".join(md_parts)

    # Fallback: pymupdf4llm sometimes returns empty for OCR-only PDFs
    # even though fitz.get_text() can read the text layer. Use plain
    # text extraction in that case.
    if not markdown_text.strip() and source_stats["total_words"] > 0:
        logger.info(
            "pymupdf4llm returned empty — falling back to fitz.get_text() "
            "(%d words across %d pages)",
            source_stats["total_words"],
            len(fallback_pages),
        )
        markdown_text = "\n\n".join(p for p in fallback_pages if p)

    return markdown_text, toc_entries, source_stats


# ---------------------------------------------------------------------------
# Regex cleanup
# ---------------------------------------------------------------------------


def clean_markdown_regex(md_text: str) -> str:
    """Apply fast heuristic fixes to common PDF conversion artifacts."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", md_text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[._\-=]{5,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text


def _comment_out_page_artifacts(
    md_text: str, total_pages: int = 0
) -> tuple[str, dict]:
    """Convert standalone page numbers and repeated running headers to HTML comments.

    Page numbers and headers are preserved as ``<!-- page: 47 -->`` or
    ``<!-- header: Preface to the Morningside Edition -->`` so they remain
    in the file as metadata but don't affect text processing.

    Returns (cleaned_text, stats_dict) where stats_dict records counts.
    """
    lines = md_text.split("\n")
    stats: dict = {"page_numbers_commented": 0, "running_headers_commented": 0}

    # --- Pass 1: detect repeated short lines (running headers) ---
    # A line that appears 3+ times, is ≤50 chars, and looks like a header
    # (not content) is likely a running header from PDF layout.
    line_counts: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if s and 3 <= len(s) <= 50:
            line_counts[s] = line_counts.get(s, 0) + 1

    def _looks_like_header(s: str) -> bool:
        """Return True if a repeated line looks like a running header, not content."""
        # Exclude dialogue attributions ("Yang Chu said:", "He replied:")
        if s.endswith(":"):
            return False
        # Exclude lines that look like sentences (contain verbs/common words)
        # Running headers are typically: titles, chapter names, author names
        # They're often italicized in markdown: _Title Here_
        if re.match(r"^_.*_$", s):
            return True  # Italicized standalone line — very likely a header
        # Exclude lines with sentence-like punctuation (periods, commas, semicolons)
        if re.search(r"[.;,!?]", s):
            return False
        # Accept remaining short standalone title-like lines
        return True

    repeated = {
        s for s, count in line_counts.items()
        if count >= 3 and _looks_like_header(s)
    }
    excluded = {
        s for s, count in line_counts.items()
        if count >= 3 and not _looks_like_header(s)
    }

    # --- Pass 2: comment out page numbers and repeated headers ---
    max_page = total_pages if total_pages else 9999
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        # Standalone page numbers: a line that is just 1-4 digits,
        # not inside a paragraph (prev and next lines are blank or absent)
        if re.match(r"^\d{1,4}$", s):
            num = int(s)
            prev_blank = (i == 0) or not lines[i - 1].strip()
            next_blank = (i == len(lines) - 1) or not lines[i + 1].strip()
            if prev_blank and next_blank and 1 <= num <= max_page:
                lines[i] = f"<!-- page: {s} -->"
                stats["page_numbers_commented"] += 1
                continue

        # Repeated running headers (only lines that passed the header filter)
        if s in repeated:
            lines[i] = f"<!-- header: {s} -->"
            stats["running_headers_commented"] += 1

    # Include detailed lists for the report
    stats["headers_commented"] = sorted(
        [(s, line_counts[s]) for s in repeated],
        key=lambda x: -x[1],
    )
    stats["headers_excluded"] = sorted(
        [(s, line_counts[s]) for s in excluded],
        key=lambda x: -x[1],
    )

    return "\n".join(lines), stats


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------


def _chunk_by_paragraphs(text: str, max_chunk_words: int) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) == 1 and len(text.split()) > max_chunk_words:
        paragraphs = text.split("\n")
        joiner = "\n"
    else:
        joiner = "\n\n"

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_words = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_chunk_words and current_chunk:
            chunks.append(joiner.join(current_chunk))
            current_chunk = []
            current_words = 0
        current_chunk.append(para)
        current_words += para_words
    if current_chunk:
        chunks.append(joiner.join(current_chunk))
    return chunks


def _chunk_by_sections(md_text: str, max_chunk_words: int = 3000) -> list[str]:
    """Split markdown into chunks that respect section boundaries.

    Splits on heading lines (^#{1,6} ). Sections that fit within
    *max_chunk_words* are grouped together. Oversized sections fall
    back to paragraph-level splitting.
    """
    heading_re = re.compile(r"^(#{1,6}\s)", re.MULTILINE)
    positions = [m.start() for m in heading_re.finditer(md_text)]

    if not positions:
        return _chunk_by_paragraphs(md_text, max_chunk_words)

    # Build section list
    sections: list[str] = []
    if positions[0] > 0:
        sections.append(md_text[: positions[0]])
    for idx, pos in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(md_text)
        sections.append(md_text[pos:end])

    if len(sections) <= 1:
        return _chunk_by_paragraphs(md_text, max_chunk_words)

    chunks: list[str] = []
    current_sections: list[str] = []
    current_words = 0

    for section in sections:
        section_words = len(section.split())

        if section_words > max_chunk_words:
            if current_sections:
                chunks.append("\n\n".join(current_sections))
                current_sections = []
                current_words = 0
            chunks.extend(_chunk_by_paragraphs(section, max_chunk_words))
            continue

        if current_words + section_words > max_chunk_words and current_sections:
            chunks.append("\n\n".join(current_sections))
            current_sections = []
            current_words = 0

        current_sections.append(section)
        current_words += section_words

    if current_sections:
        chunks.append("\n\n".join(current_sections))

    return chunks


# ---------------------------------------------------------------------------
# LLM cleanup (Anthropic Claude)
# ---------------------------------------------------------------------------


def _parse_llm_response(raw: str) -> tuple[str, dict]:
    """Split an LLM response into (cleaned_text, issues_dict).

    The LLM is asked to prefix its response with a ISSUES_JSON: line.
    If parsing fails, treat the entire response as cleaned text.
    """
    issues: dict = {"issues": [], "garbled_words_fixed": 0}
    if raw.startswith("ISSUES_JSON:"):
        first_nl = raw.find("\n")
        if first_nl == -1:
            return raw, issues
        json_line = raw[len("ISSUES_JSON:") : first_nl].strip()
        cleaned = raw[first_nl:].lstrip("\n")
        try:
            issues = json.loads(json_line)
        except (json.JSONDecodeError, ValueError):
            pass
        return cleaned, issues
    return raw, issues


def clean_markdown_llm(
    md_text: str, report: ConversionReport | None = None
) -> str:
    """Use Anthropic Claude to clean up conversion artifacts in the markdown."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM cleanup")
        return md_text

    client = Anthropic(api_key=api_key)

    max_chunk_words = 3000
    chunks = _chunk_by_sections(md_text, max_chunk_words)

    # Pre-compute line offsets for each chunk so the report can show line ranges
    line_offset = 1  # 1-indexed
    chunk_line_starts: list[int] = []
    for chunk in chunks:
        chunk_line_starts.append(line_offset)
        line_offset += chunk.count("\n") + 1  # +1 for the join separator

    cleaned_chunks: list[str] = []
    min_retention = 0.6
    api_failed = False  # stop retrying after a fatal API error

    for i, chunk in enumerate(chunks):
        input_words = len(chunk.split())
        chunk_lines = chunk.count("\n") + 1
        start_line = chunk_line_starts[i]
        end_line = start_line + chunk_lines - 1

        if api_failed:
            cleaned_chunks.append(chunk)
            if report is not None:
                report.chunks_fell_back += 1
                report.chunk_details.append({
                    "chunk_num": i + 1,
                    "status": "skipped",
                    "reason": "Skipped after earlier fatal API error",
                    "start_line": start_line,
                    "end_line": end_line,
                    "words": input_words,
                })
            continue

        logger.info(
            "LLM cleanup: chunk %d/%d (%d words, lines %d–%d)",
            i + 1, len(chunks), input_words, start_line, end_line,
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=8192,
                system=CLEANUP_PROMPT,
                messages=[{"role": "user", "content": chunk}],
                temperature=0,
            )
            raw_response = response.content[0].text
        except Exception as exc:
            exc_str = str(exc)
            # Content filtering is chunk-specific — skip this chunk, keep going.
            # Auth / credit errors are fatal — stop retrying.
            is_content_filter = "content filtering" in exc_str.lower()
            if is_content_filter:
                reason = "Blocked by content filter"
                logger.warning(
                    "LLM cleanup: chunk %d/%d blocked by content filter — keeping original",
                    i + 1,
                    len(chunks),
                )
            else:
                reason = f"API error: {exc}"
                logger.warning(
                    "LLM cleanup failed on chunk %d/%d: %s — skipping LLM for remaining chunks",
                    i + 1,
                    len(chunks),
                    exc,
                )
                api_failed = True

            cleaned_chunks.append(chunk)
            if report is not None:
                report.chunk_issues.append(
                    {"issues": [f"LLM cleanup unavailable: {exc}"], "garbled_words_fixed": 0}
                )
                report.chunks_fell_back += 1
                report.chunk_details.append({
                    "chunk_num": i + 1,
                    "status": "failed",
                    "reason": reason,
                    "start_line": start_line,
                    "end_line": end_line,
                    "words": input_words,
                })
            continue

        cleaned, chunk_issues = _parse_llm_response(raw_response)

        output_words = len(cleaned.split())
        retention = output_words / input_words if input_words else 1.0

        if retention < min_retention:
            logger.warning(
                "Chunk %d/%d: LLM dropped too much content "
                "(%d -> %d words, %.0f%% retention) — keeping original",
                i + 1,
                len(chunks),
                input_words,
                output_words,
                retention * 100,
            )
            cleaned_chunks.append(chunk)
            if report is not None:
                report.chunks_fell_back += 1
                report.chunk_details.append({
                    "chunk_num": i + 1,
                    "status": "failed",
                    "reason": f"Low retention ({retention*100:.0f}%)",
                    "start_line": start_line,
                    "end_line": end_line,
                    "words": input_words,
                })
        else:
            cleaned_chunks.append(cleaned)
            if report is not None:
                report.chunk_details.append({
                    "chunk_num": i + 1,
                    "status": "cleaned",
                    "reason": "",
                    "start_line": start_line,
                    "end_line": end_line,
                    "words": input_words,
                })

        if report is not None:
            report.chunk_issues.append(chunk_issues)

    return "\n\n".join(cleaned_chunks)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _is_structural(line: str) -> bool:
    """Check if a line is a heading, illustration, list item, or other structural element."""
    s = line.strip()
    if not s:
        return True
    return (
        s.startswith("#")
        or s.startswith("[Illustration")
        or s.startswith("![Illustration")
        or s.startswith("- ")
        or s.startswith("* ")
        or s.startswith("|")
        or s.startswith("***")
        or s.startswith("---")
    )


def _find_toc_matches(text: str, toc_entries: list | None = None) -> list[dict]:
    """Find body lines that match TOC entries and return match info for the report.

    Does NOT modify the text — just identifies potential heading locations
    so the user can decide whether to promote them.

    Returns a list of dicts: {"toc_title": str, "line_num": int, "line_text": str}
    """
    if not toc_entries:
        return []

    lines = text.split("\n")

    # Build lookup: normalised title -> original title
    toc_lookup: dict[str, str] = {}
    for _level, title, _page in toc_entries:
        key = re.sub(r"[_*`#\[\]()\\]", "", title).strip().lower()
        key = re.sub(r"\s+", " ", key)
        if len(key) > 2:
            toc_lookup[key] = title.strip()

    def _clean_line(raw: str) -> str:
        """Strip markdown emphasis, numbering prefix, and whitespace."""
        s = raw.strip()
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"[_*`\\]", "", s)
        s = re.sub(r"^\d+\.\s*", "", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    matches: list[dict] = []
    found_keys: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue

        cleaned = _clean_line(stripped)
        if not cleaned:
            continue

        match_key = None
        if cleaned in toc_lookup:
            match_key = cleaned
        else:
            for toc_key in toc_lookup:
                toc_cleaned = re.sub(r"^\d+\.\s*", "", toc_key).strip()
                if cleaned == toc_cleaned and toc_key not in found_keys:
                    match_key = toc_key
                    break

        if match_key and match_key not in found_keys:
            matches.append({
                "toc_title": toc_lookup[match_key],
                "line_num": i + 1,  # 1-indexed
                "line_text": stripped,
            })
            found_keys.add(match_key)

    return matches


def _normalize_headings(text: str, toc_entries: list | None = None) -> str:
    """Placeholder — heading normalization is intentionally conservative.

    Returns the text unchanged. TOC match info is gathered separately
    by _find_toc_matches() and included in the report so the user can
    decide whether to promote headings manually.
    """
    return text


def _join_broken_paragraphs(text: str) -> str:
    """Join single-line paragraphs that look like hard-wrapped prose."""
    MIN_WRAP_LENGTH = 55

    paragraphs = re.split(r"\n\n", text)
    result: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()

        if not para or "\n" in para or _is_structural(para):
            result.append(paragraphs[i])
            i += 1
            continue

        joined = para
        while i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if not next_para or "\n" in next_para or _is_structural(next_para):
                break
            if joined.rstrip()[-1:] in ".?!:\"'":
                break
            last_line = joined.rsplit("\n", 1)[-1] if "\n" in joined else joined
            if len(last_line) < MIN_WRAP_LENGTH:
                break
            if not next_para[0].islower():
                break
            i += 1
            joined = joined + " " + next_para

        result.append(joined)
        i += 1

    return "\n\n".join(result)


def normalize_markdown(md_text: str, toc_entries: list | None = None) -> str:
    """Post-LLM normalization pass for cross-chunk consistency."""
    text = md_text
    text = re.sub(r"\\([.()[\]*_~`])", r"\1", text)
    text = re.sub(r"!\[Illustration:", "[Illustration:", text)
    text = _normalize_headings(text, toc_entries=toc_entries)
    text = _join_broken_paragraphs(text)
    logger.info("After normalization: %d chars, %d words", len(text), len(text.split()))
    return text


# ---------------------------------------------------------------------------
# Content loss detection
# ---------------------------------------------------------------------------


def is_garbled(md_text: str) -> bool:
    """Detect if text is mostly garbled (font encoding failures, bad OCR)."""
    words = md_text.split()
    if not words:
        return True
    single_chars = sum(1 for w in words if len(w) == 1 and w not in ("I", "a", "A"))
    ratio = single_chars / len(words)
    return ratio > 0.3


def _check_content_coverage(
    source_stats: dict,
    toc_entries: list,
    output_md: str,
) -> dict:
    """Compare source PDF stats against final output to detect content loss."""
    output_words = len(output_md.split())
    source_words = source_stats["total_words"]
    retention = output_words / source_words if source_words else 1.0

    output_lower = output_md.lower()
    missing_toc: list[str] = []
    for _level, title, _page in toc_entries:
        normalized = title.strip().lower()
        if len(normalized) > 3 and normalized not in output_lower:
            missing_toc.append(title)

    if retention >= 0.95 and not missing_toc:
        confidence = "high"
    elif retention >= 0.65 and len(missing_toc) <= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "source_words": source_words,
        "output_words": output_words,
        "retention_pct": round(retention * 100, 1),
        "missing_toc_entries": missing_toc,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Front/back matter boundary detection
# ---------------------------------------------------------------------------

_DEFAULT_FRONT_MATTER = (
    "Everything before the first chapter or section of the primary text: "
    "title pages, copyright pages, tables of contents, editor/translator "
    "introductions, prefaces, dedications, forewords, bibliographic notes, "
    "acknowledgments, lists of abbreviations"
)

_DEFAULT_BACK_MATTER = (
    "Everything after the last chapter or section of the primary text: "
    "appendices, indexes, glossaries, endnotes, bibliographies, editorial "
    "afterwords, colophons, about the author sections"
)

BOUNDARY_PROMPT_TEMPLATE = """\
You are analyzing a document to identify where the front matter ends and \
where the back matter begins, so that researchers can search only the \
primary text when needed.

FRONT MATTER is defined as: {front_matter_def}

BACK MATTER is defined as: {back_matter_def}

PRIMARY TEXT is everything between the front matter and back matter — the \
main body of the work.

You will be given the first ~200 lines and last ~200 lines of the document. \
Analyze them and respond with a JSON object in this exact format:

{{"front_matter_end_line": <line number of the LAST line of front matter, or null if none>, \
"front_matter_summary": "<brief description of what the front matter contains>", \
"back_matter_start_line": <line number of the FIRST line of back matter, or null if none>, \
"back_matter_summary": "<brief description of what the back matter contains>"}}

Line numbers are 1-indexed. If the document has no discernible front or back \
matter, use null for that field.

Respond with ONLY the JSON object, nothing else."""


def _detect_matter_boundaries(
    md_text: str, report: ConversionReport
) -> str:
    """Detect front/back matter and insert HTML comment markers.

    Uses an LLM to analyze the beginning and end of the document,
    then inserts <!-- FRONT_MATTER_END --> and <!-- BACK_MATTER_START -->
    markers at the detected boundaries.

    Returns the text with markers inserted.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return md_text

    lines = md_text.split("\n")
    if len(lines) < 20:
        return md_text

    # Build a context window: first ~200 and last ~200 lines
    head_size = min(200, len(lines))
    tail_size = min(200, len(lines))
    head = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:head_size]))
    tail_start = max(head_size, len(lines) - tail_size)
    tail = "\n".join(
        f"{i+1}: {l}" for i, l in enumerate(lines[tail_start:], start=tail_start)
    )

    context = f"=== FIRST {head_size} LINES ===\n{head}"
    if tail_start > head_size:
        context += f"\n\n=== LAST {tail_size} LINES (starting at line {tail_start + 1}) ===\n{tail}"

    front_def = os.environ.get("FRONT_MATTER_DEFINITION", _DEFAULT_FRONT_MATTER)
    back_def = os.environ.get("BACK_MATTER_DEFINITION", _DEFAULT_BACK_MATTER)
    system_prompt = BOUNDARY_PROMPT_TEMPLATE.format(
        front_matter_def=front_def, back_matter_def=back_def
    )

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
            temperature=0,
        )
        raw = response.content[0].text.strip()
        # Try direct parse first; fall back to extracting JSON from the response
        try:
            boundaries = json.loads(raw)
        except json.JSONDecodeError:
            # LLM may wrap JSON in ```json ... ``` or add commentary
            json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if json_match:
                boundaries = json.loads(json_match.group())
            else:
                logger.warning("Failed to extract JSON from boundary response: %s", raw[:200])
                return md_text
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        logger.warning("Failed to parse boundary detection response: %s", exc)
        return md_text
    except Exception:
        logger.exception("Boundary detection LLM call failed")
        return md_text

    front_end = boundaries.get("front_matter_end_line")
    back_start = boundaries.get("back_matter_start_line")

    # Validate line numbers
    if front_end is not None:
        if not isinstance(front_end, int) or front_end < 1 or front_end >= len(lines):
            front_end = None
    if back_start is not None:
        if not isinstance(back_start, int) or back_start < 1 or back_start > len(lines):
            back_start = None
    if front_end and back_start and front_end >= back_start:
        logger.warning(
            "Boundary detection: front_end (%d) >= back_start (%d) — skipping",
            front_end,
            back_start,
        )
        return md_text

    # Populate report
    report.front_matter_end_line = front_end
    report.front_matter_summary = boundaries.get("front_matter_summary", "")
    report.back_matter_start_line = back_start
    report.back_matter_summary = boundaries.get("back_matter_summary", "")

    # Insert markers (insert from bottom up to preserve line numbers)
    if back_start is not None:
        lines.insert(back_start - 1, "\n<!-- BACK_MATTER_START -->\n")
    if front_end is not None:
        lines.insert(front_end, "\n<!-- FRONT_MATTER_END -->\n")

    logger.info(
        "Boundary markers inserted: front_end=%s, back_start=%s",
        front_end,
        back_start,
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def convert_and_clean(
    md_text: str,
    report: ConversionReport,
    toc_entries: list | None = None,
    total_pages: int = 0,
) -> str:
    """Run the full cleanup pipeline: regex -> LLM -> normalize -> comment artifacts."""
    report.per_stage_words["raw"] = len(md_text.split())

    md_text = clean_markdown_regex(md_text)
    report.per_stage_words["after_regex"] = len(md_text.split())

    if is_garbled(md_text):
        logger.warning("Text appears garbled — skipping LLM cleanup")
        return ""

    md_text = clean_markdown_llm(md_text, report=report)
    report.per_stage_words["after_llm"] = len(md_text.split())

    md_text = normalize_markdown(md_text, toc_entries=toc_entries)
    report.per_stage_words["after_normalize"] = len(md_text.split())

    md_text, artifact_stats = _comment_out_page_artifacts(md_text, total_pages)
    report.page_artifact_stats = artifact_stats
    report.per_stage_words["after_artifact_cleanup"] = len(md_text.split())

    return md_text


def convert_docx_bytes(docx_bytes: bytes) -> str:
    """Convert in-memory DOCX bytes to a Markdown string."""
    result = mammoth.convert_to_markdown(io.BytesIO(docx_bytes))
    return result.value


def convert_txt_bytes(txt_bytes: bytes) -> str:
    """Convert in-memory TXT bytes to a Markdown string."""
    return txt_bytes.decode("utf-8", errors="replace")


def convert_file_bytes(
    file_bytes: bytes, filename: str, filetype: str
) -> tuple[str, ConversionReport]:
    """Convert in-memory file bytes to a cleaned Markdown string.

    Returns (markdown_text, report).
    """
    report = ConversionReport(filename=filename, filetype=filetype)

    toc_entries: list = []
    source_stats: dict = {"per_page_words": [], "total_words": 0}

    if filetype == "pdf":
        # OCR pre-processing
        pdf_bytes, ocr_info = _ocr_pdf_if_needed(file_bytes)
        report.ocr_needed = ocr_info["needed"]
        report.ocr_pages = ocr_info["pages_ocrd"]
        report.total_pages = ocr_info["total_pages"]

        # Structure-aware extraction
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            md_text, toc_entries, source_stats = _extract_pdf_with_structure(tmp_path)
        finally:
            Path(tmp_path).unlink()

        report.toc_entries_total = len(toc_entries)
        report.source_words = source_stats["total_words"]

    elif filetype == "docx":
        md_text = convert_docx_bytes(file_bytes)
        report.source_words = len(md_text.split())

    elif filetype == "text":
        md_text = convert_txt_bytes(file_bytes)
        report.source_words = len(md_text.split())

    else:
        raise ValueError(f"Unsupported file type: {filetype}")

    if not md_text.strip():
        report.output_words = 0
        report.retention_pct = 0.0
        report.confidence = "low"
        return md_text, report

    logger.info("Raw conversion: %d chars, %d words", len(md_text), len(md_text.split()))
    cleaned = convert_and_clean(
        md_text, report=report, toc_entries=toc_entries,
        total_pages=report.total_pages,
    )
    logger.info("After cleanup: %d chars, %d words", len(cleaned), len(cleaned.split()))

    # Detect and mark front/back matter boundaries
    if cleaned.strip():
        cleaned = _detect_matter_boundaries(cleaned, report)

    # Find TOC matches for heading suggestions in the report
    if toc_entries and cleaned.strip():
        report.heading_suggestions = _find_toc_matches(cleaned, toc_entries)

    # Content loss detection (PDFs with TOC)
    if filetype == "pdf" and source_stats["total_words"] > 0:
        coverage = _check_content_coverage(source_stats, toc_entries, cleaned)
        report.output_words = coverage["output_words"]
        report.retention_pct = coverage["retention_pct"]
        report.missing_toc_entries = coverage["missing_toc_entries"]
        report.confidence = coverage["confidence"]
    else:
        report.output_words = len(cleaned.split())
        report.retention_pct = (
            round(report.output_words / report.source_words * 100, 1)
            if report.source_words
            else 100.0
        )
        report.confidence = "high" if report.retention_pct >= 95 else "medium"

    # Prepend YAML frontmatter for library cataloging
    if cleaned.strip():
        cleaned = _build_yaml_frontmatter(report) + cleaned

    return cleaned, report


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------


def _build_yaml_frontmatter(report: ConversionReport) -> str:
    """Build a YAML frontmatter block from the conversion report.

    This metadata helps when building a library of converted .md files —
    it's machine-readable and compatible with tools like Obsidian, Jekyll,
    and Pandoc.
    """
    from datetime import date

    lines = ["---"]
    lines.append(f"source_file: \"{report.filename}\"")
    lines.append(f"source_type: {report.filetype}")
    lines.append(f"converted: {date.today().isoformat()}")
    lines.append(f"confidence: {report.confidence}")
    lines.append(f"retention_pct: {report.retention_pct}")
    lines.append(f"source_words: {report.source_words}")
    lines.append(f"output_words: {report.output_words}")

    if report.filetype == "pdf":
        lines.append(f"pages: {report.total_pages}")
        if report.ocr_needed:
            lines.append(f"ocr_applied: true")
            lines.append(f"ocr_pages: {len(report.ocr_pages)}")
        else:
            lines.append(f"ocr_applied: false")

    if report.toc_entries_total > 0:
        lines.append(f"toc_entries: {report.toc_entries_total}")

    if report.front_matter_end_line is not None:
        lines.append(f"front_matter_ends: {report.front_matter_end_line}")
    if report.back_matter_start_line is not None:
        lines.append(f"back_matter_starts: {report.back_matter_start_line}")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output splitting (for platforms with character limits like HackMD)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def split_markdown(md_text: str, max_chars: int) -> list[str]:
    """Split markdown into parts under *max_chars*, breaking at headings.

    Prefers to split at heading boundaries. Falls back to paragraph
    boundaries if no heading is found before the limit. Returns a list
    of parts, each under max_chars (unless a single paragraph exceeds it,
    in which case that paragraph becomes its own part).
    """
    if len(md_text) <= max_chars:
        return [md_text]

    parts: list[str] = []
    remaining = md_text

    while len(remaining) > max_chars:
        # Look for the last heading before the limit
        window = remaining[:max_chars]
        split_pos = None

        # Find last heading in window
        for m in _HEADING_RE.finditer(window):
            # Split just before the heading (at the preceding newline)
            pos = m.start()
            if pos > 0:
                split_pos = pos

        # Fall back to last paragraph break
        if split_pos is None or split_pos < max_chars // 4:
            last_para = window.rfind("\n\n")
            if last_para > max_chars // 4:
                split_pos = last_para + 1  # keep one newline at end

        # Last resort: hard split at limit
        if split_pos is None or split_pos < max_chars // 4:
            split_pos = max_chars

        parts.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip("\n")

    if remaining.strip():
        parts.append(remaining)

    return parts


# Keep backward-compatible alias
def convert_pdf_bytes(
    pdf_bytes: bytes, filename: str = "document.pdf"
) -> tuple[str, ConversionReport]:
    """Convert in-memory PDF bytes to a cleaned Markdown string."""
    return convert_file_bytes(pdf_bytes, filename, "pdf")


def convert_pdfs(pdf_dir: Path = Path("pdfs/"), out_dir: Path = Path("markdown/")):
    """Convert all PDFs in pdf_dir to Markdown files in out_dir."""
    out_dir.mkdir(exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    max_chars_str = os.environ.get("MAX_OUTPUT_CHARS", "")
    max_chars = int(max_chars_str) if max_chars_str.strip().isdigit() else 0

    for pdf_file in pdf_files:
        md_text, report = convert_file_bytes(
            pdf_file.read_bytes(), pdf_file.name, "pdf"
        )

        if max_chars and len(md_text) > max_chars:
            parts = split_markdown(md_text, max_chars)
            total = len(parts)
            for idx, part in enumerate(parts, 1):
                part_path = out_dir / f"{pdf_file.stem}_part{idx}of{total}.md"
                part_path.write_text(part)
                print(f"  Part {idx}/{total}: {part_path} ({len(part):,} chars)")
        else:
            out_path = out_dir / f"{pdf_file.stem}.md"
            out_path.write_text(md_text)
            print(f"Converted: {pdf_file.name} -> {out_path}")

        report_path = out_dir / f"{pdf_file.stem}_report.md"
        report_path.write_text(report.to_markdown())

        print(f"  Report: {report_path}")
        print(f"  Confidence: {report.confidence} | Retention: {report.retention_pct}%")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    convert_pdfs()
