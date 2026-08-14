import os
import sys
from pathlib import Path

# Make the synth helpers importable as a plain module from any test.
sys.path.insert(0, str(Path(__file__).parent))

# Qt tests must never try to open a real display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
