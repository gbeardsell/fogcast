"""June 14, 2026, 7 a.m. PDT: run Model I on the included historical inputs."""
import sys
from pathlib import Path

from fogcast.residual_unet_fogcast.forecast import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main([
        "--assets", str(root / "assets/model-i"),
        "--offline-inputs", str(root / "assets/historical-20260614T140000Z/inputs.npz"),
        "--output", str(root / "outputs/historical"),
        *sys.argv[1:],
    ])
