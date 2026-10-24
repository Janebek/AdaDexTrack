"""Train one offline BC tracker from the selected specialists' demonstrations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adadextrack.bc import main

if __name__ == "__main__":
    sys.exit(main())
