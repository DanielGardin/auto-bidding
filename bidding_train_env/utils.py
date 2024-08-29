from typing import Protocol, Optional

from pathlib import Path
from functools import partial

def get_root_path():
    return Path(__file__).parent.parent
