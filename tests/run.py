# Minimal test runner (no pytest dependency).
# Usage: python tests/run.py
from __future__ import annotations

import importlib
import os
import sys
import traceback

TESTS = ("test_layers", "test_read", "test_acceptance", "test_evolve")


def main() -> int:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(pkg_dir))
    failed = 0
    passed = 0
    for name in TESTS:
        mod = importlib.import_module("tests." + name)
        for fn in sorted(dir(mod)):
            if not fn.startswith("test_"):
                continue
            try:
                getattr(mod, fn)()
                passed += 1
                print("PASS tests.%s.%s" % (name, fn))
            except Exception:
                failed += 1
                print("FAIL tests.%s.%s" % (name, fn))
                traceback.print_exc()
    status = "OK" if failed == 0 else "FAILED"
    print("%s: %d/%d passed" % (status, passed, passed + failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
