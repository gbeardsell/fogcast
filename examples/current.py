"""Fetch current GOES-18 and NBM data, then run the Model I three-seed blend."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fogcast.forecast import main

if __name__ == "__main__":
    main([
        "--assets", str(ROOT / "assets/model-i"),
        "--output", str(ROOT / "outputs/current"),
        *sys.argv[1:],
    ])
