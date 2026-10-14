"""Minimal test runner (pytest is optional): python run_tests.py [-k substring]"""
import importlib
import os
import sys
import traceback
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main(argv):
    pattern = argv[argv.index("-k") + 1] if "-k" in argv else ""
    passed = failed = skipped = 0
    for fname in sorted(os.listdir(os.path.join(ROOT, "tests"))):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        mod = importlib.import_module(fname[:-3])
        for name in sorted(dir(mod)):
            if not name.startswith("test_") or pattern not in name:
                continue
            try:
                getattr(mod, name)()
                passed += 1
                print(f"PASS  {fname}::{name}")
            except unittest.SkipTest as e:
                skipped += 1
                print(f"SKIP  {fname}::{name}  ({e})")
            except Exception:
                failed += 1
                print(f"FAIL  {fname}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
