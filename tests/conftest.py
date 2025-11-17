import sys
from pathlib import Path

# Ensure the source package is importable without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - defensive
    sys.path.insert(0, str(ROOT))
