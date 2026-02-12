#!/usr/bin/env python3
"""Convert documents (PDF, DOCX, TXT) to Markdown — used by both the CLI and Slack bot."""

import io
import logging
import os
import re
import tempfile
from pathlib import Path

import mammoth
import pymupdf4llm
from openai import OpenAI

logger = logging.getLogger(__name__)

CLEANUP_PROMPT = """\
You are a document cleanup assistant. Your HIGHEST PRIORITY is preserving every \
piece of original content. The following markdown was extracted from a document \
and may contain conversion artifacts.

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

Return ONLY the cleaned markdown, nothing else."""


def clean_markdown_regex(md_text: str) -> str:
    """Apply fast heuristic fixes to common PDF conversion artifacts."""
    # Strip null / control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", md_text)
    # Rejoin words broken by a hyphen at end of line
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are only repeated punctuation artifacts (e.g. "....." or "____")
    text = re.sub(r"^[._\-=]{5,}$", "", text, flags=re.MULTILINE)
    # Normalize trailing whitespace on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text


def clean_markdown_llm(md_text: str) -> str:
    """Use OpenAI to clean up conversion artifacts in the markdown."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM cleanup")
        return md_text

    client = OpenAI(api_key=api_key)

    # Process in chunks by splitting on paragraph boundaries to preserve formatting
    max_chunk_words = 3000
    words = md_text.split()
    if len(words) <= max_chunk_words:
        chunks = [md_text]
    else:
        # Try paragraph boundaries first, fall back to single newlines for plain text
        paragraphs = re.split(r"\n\n+", md_text)
        if len(paragraphs) == 1 and len(words) > max_chunk_words:
            paragraphs = md_text.split("\n")
            joiner = "\n"
        else:
            joiner = "\n\n"
        chunks = []
        current_chunk = []
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

    cleaned_chunks = []
    min_retention = 0.6  # if LLM returns <60% of input words, keep original chunk
    for i, chunk in enumerate(chunks):
        input_words = len(chunk.split())
        logger.info("LLM cleanup: chunk %d/%d (%d words)", i + 1, len(chunks), input_words)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLEANUP_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=0,
        )
        cleaned = response.choices[0].message.content
        output_words = len(cleaned.split())
        retention = output_words / input_words if input_words else 1.0
        if retention < min_retention:
            logger.warning(
                "Chunk %d/%d: LLM dropped too much content (%d → %d words, %.0f%% retention) — keeping original",
                i + 1, len(chunks), input_words, output_words, retention * 100,
            )
            cleaned_chunks.append(chunk)
        else:
            cleaned_chunks.append(cleaned)

    return "\n\n".join(cleaned_chunks)


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


def _normalize_headings(text: str) -> str:
    """Detect TOC entries and ensure body section headings match their format."""
    toc_pattern = re.compile(r"^###\s+(\d+)\.\s+(.+)$", re.MULTILINE)

    # Phase 1: Find the TOC — a block of 5+ consecutive ### N. headings
    lines = text.split("\n")
    toc_entries = {}  # num_str -> title
    toc_end_line = None
    consecutive = 0
    block_start = None

    for i, line in enumerate(lines):
        if toc_pattern.match(line):
            if block_start is None:
                block_start = i
            consecutive += 1
        else:
            if consecutive >= 5:
                toc_end_line = i
                break
            consecutive = 0
            block_start = None

    if toc_end_line is None:
        return text

    for i in range(block_start, toc_end_line):
        m = toc_pattern.match(lines[i])
        if m:
            toc_entries[m.group(1)] = m.group(2).strip()

    if not toc_entries:
        return text

    num_set = set(toc_entries.keys())

    # Phase 2: After the TOC, find and normalize body section headings
    # Match lines like "N. Title", "N\. _Title_", etc. that should be ### headings
    body_heading = re.compile(r"^(\d+)\\?\.\s+_?(.+?)_?\s*$")

    for i in range(toc_end_line, len(lines)):
        line = lines[i].strip()
        # Skip lines already in correct format
        if toc_pattern.match(line):
            continue
        # Skip long lines (body paragraphs, not headings)
        if len(line) > 200:
            continue
        m = body_heading.match(line)
        if m and m.group(1) in num_set:
            num = m.group(1)
            lines[i] = f"### {num}. {toc_entries[num]}"

    return "\n".join(lines)


def _join_broken_paragraphs(text: str) -> str:
    """Join single-line paragraphs that look like hard-wrapped prose.

    Conservative: only joins when a line is long enough to suggest it hit a
    fixed-width wrap limit AND the next line starts lowercase (continuation).
    Short lines are left alone to preserve verse, dialogue, and lists.
    """
    MIN_WRAP_LENGTH = 55  # lines shorter than this are likely intentional

    paragraphs = re.split(r"\n\n", text)
    result = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()

        # Skip structural or multi-line paragraphs
        if not para or "\n" in para or _is_structural(para):
            result.append(paragraphs[i])
            i += 1
            continue

        # Accumulate single-line continuations
        joined = para
        while i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            # Stop if next is empty, structural, or multi-line
            if not next_para or "\n" in next_para or _is_structural(next_para):
                break
            # Stop if current line ends with sentence-terminal punctuation
            if joined.rstrip()[-1:] in ".?!:\"'":
                break
            # Only join if the current line is long enough to be a hard wrap
            # AND the next line starts lowercase (strong continuation signal)
            last_line = joined.rsplit("\n", 1)[-1] if "\n" in joined else joined
            if len(last_line) < MIN_WRAP_LENGTH:
                break
            if not next_para[0].islower():
                break
            # Join the continuation
            i += 1
            joined = joined + " " + next_para

        result.append(joined)
        i += 1

    return "\n\n".join(result)


def normalize_markdown(md_text: str) -> str:
    """Post-LLM normalization pass for cross-chunk consistency."""
    text = md_text

    # 1. Remove unnecessary backslash escapes from conversion
    text = re.sub(r"\\([.()[\]*_~`])", r"\1", text)

    # 2. Normalize illustration markup to consistent [Illustration: ...] format
    text = re.sub(r"!\[Illustration:", "[Illustration:", text)

    # 3. Normalize section headings to match TOC
    text = _normalize_headings(text)

    # 4. Join broken paragraphs (single lines that should be continuous)
    text = _join_broken_paragraphs(text)

    logger.info("After normalization: %d chars, %d words", len(text), len(text.split()))
    return text


def is_garbled(md_text: str) -> bool:
    """Detect if text is mostly garbled (font encoding failures, bad OCR)."""
    # Count single-character "words" separated by spaces/punctuation
    words = md_text.split()
    if not words:
        return True
    single_chars = sum(1 for w in words if len(w) == 1 and w not in ("I", "a", "A"))
    ratio = single_chars / len(words)
    return ratio > 0.3


def convert_and_clean(md_text: str) -> str:
    """Run the full cleanup pipeline: regex, then LLM, then normalize."""
    md_text = clean_markdown_regex(md_text)
    if is_garbled(md_text):
        logger.warning("Text appears garbled — skipping LLM cleanup")
        return ""
    md_text = clean_markdown_llm(md_text)
    md_text = normalize_markdown(md_text)
    return md_text


def convert_docx_bytes(docx_bytes: bytes) -> str:
    """Convert in-memory DOCX bytes to a Markdown string."""
    result = mammoth.convert_to_markdown(io.BytesIO(docx_bytes))
    return result.value


def convert_txt_bytes(txt_bytes: bytes) -> str:
    """Convert in-memory TXT bytes to a Markdown string."""
    return txt_bytes.decode("utf-8", errors="replace")


def convert_file_bytes(file_bytes: bytes, filename: str, filetype: str) -> str:
    """Convert in-memory file bytes to a cleaned Markdown string."""
    if filetype == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            md_text = pymupdf4llm.to_markdown(tmp_path)
        finally:
            Path(tmp_path).unlink()
    elif filetype == "docx":
        md_text = convert_docx_bytes(file_bytes)
    elif filetype == "text":
        md_text = convert_txt_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")

    if not md_text.strip():
        return md_text
    logger.info("Raw conversion: %d chars, %d words", len(md_text), len(md_text.split()))
    cleaned = convert_and_clean(md_text)
    logger.info("After cleanup: %d chars, %d words", len(cleaned), len(cleaned.split()))
    return cleaned


# Keep backward-compatible alias
def convert_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """Convert in-memory PDF bytes to a cleaned Markdown string."""
    return convert_file_bytes(pdf_bytes, filename, "pdf")


def convert_pdfs(pdf_dir: Path = Path("pdfs/"), out_dir: Path = Path("markdown/")):
    """Convert all PDFs in pdf_dir to Markdown files in out_dir."""
    out_dir.mkdir(exist_ok=True)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    for pdf_file in pdf_files:
        md_text = pymupdf4llm.to_markdown(str(pdf_file))
        md_text = convert_and_clean(md_text)
        out_path = out_dir / f"{pdf_file.stem}.md"
        out_path.write_text(md_text)
        print(f"Converted: {pdf_file.name} -> {out_path}")


if __name__ == "__main__":
    convert_pdfs()
