"""Fetch current GOES-18 and NBM data, then run the Model I three-seed blend."""
import sys
from pathlib import Path

from fogcast.forecast import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main([
        "--assets", str(root / "assets/model-i"),
        "--output", str(root / "outputs/current"),
        *sys.argv[1:],
    ])
