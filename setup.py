"""
Setup script for Bangalore Adaptive Traffic Signal Control
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="bangalore-traffic-rl",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control on Bangalore Intersections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bangalore-traffic-rl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Traffic Simulation",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.2",
            "myst-parser>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-train=scripts.04_train_curriculum:main",
            "traffic-evaluate=scripts.05_evaluate:main",
            "traffic-visualize=scripts.06_visualize:main",
            "traffic-demo=scripts.07_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
