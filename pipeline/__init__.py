"""Camera-to-LaTeX pipeline: capture -> segment -> classify -> layout -> PDF."""
import sys
from pathlib import Path

# Let "python -m pipeline.x" and the tools find config.py at the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
