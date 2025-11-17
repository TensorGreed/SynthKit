from pathlib import Path
from io import StringIO
from pdfminer.high_level import extract_text_to_fp


def read_pdf(path: Path) -> str:
    output = StringIO()
    with path.open("rb") as f:
        extract_text_to_fp(f, output)
    return output.getvalue()
