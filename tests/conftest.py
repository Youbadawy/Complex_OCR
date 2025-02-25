import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)
