"""Pytest setup.

The source packages use numbered directory names (``01_basics`` ...) which are
not importable as Python packages, so their directories are added to
``sys.path`` here, letting tests import the modules by bare name.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("01_basics", "02_detection", "03_anomaly", "04_pipeline"):
    sys.path.insert(0, os.path.join(_REPO, _sub))
