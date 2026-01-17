#!/usr/bin/env python
"""
Run comparison script from project root.
Usage: python run_compare.py
"""
import sys
import os

# Change to project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

if __name__ == "__main__":
    import runpy
    sys.argv[0] = os.path.join(project_root, "src", "evaluation", "compare.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
