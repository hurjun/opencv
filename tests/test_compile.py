"""Guard against import/syntax regressions across every source file.

The original detection drivers raised ``SyntaxError`` because they imported
digit-prefixed package names. Compiling every module keeps that from regressing
without requiring the (heavy) torch import.
"""

import glob
import os
import py_compile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRS = ("01_basics", "02_detection", "03_anomaly", "04_pipeline")


def test_all_sources_compile() -> None:
    files = []
    for sub in _DIRS:
        files += glob.glob(os.path.join(_REPO, sub, "*.py"))
    assert files, "no source files found"
    for path in files:
        py_compile.compile(path, doraise=True)
