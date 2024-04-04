"""Initial module for miniTransformer package."""

from os.path import isfile, dirname

version_file = f"{dirname(__file__)}/version.txt"

if isfile(version_file):
    with open(version_file, encoding="utf-8") as version_file:
        __version__ = version_file.read().strip()
