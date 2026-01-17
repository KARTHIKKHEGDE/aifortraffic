#!/usr/bin/env python
"""
Run training script from project root.
Usage: python run_train.py [args]
"""
import sys
import os

# Change to project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Import and run the training script
from src.training import train_ppo

if __name__ == "__main__":
    # Execute main from train_ppo
    import runpy
    sys.argv[0] = os.path.join(project_root, "src", "training", "train_ppo.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
