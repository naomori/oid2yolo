"""Minimal setup file for oid2yolo project."""
import re

from setuptools import setup

with open("src/oid2yolo/__init__.py", encoding="utf8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="Oid2yolo",
    version=version,
    install_requires=[
        "PyYAML",
        "pandas"
    ]
)
