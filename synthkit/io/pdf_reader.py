"""Thin wrappers around pdfminer for extracting plain text."""

from pathlib import Path
from io import StringIO

from pdfminer.high_level import extract_text_to_fp


def read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pdfminer."""
    output = StringIO()
    with path.open("rb") as handle:
        extract_text_to_fp(handle, output)
    return output.getvalue()
