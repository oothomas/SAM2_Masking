"""
sam2_masking
============

Command‑line utilities that wrap Facebook SAMURAI (SAM‑2) for object‑centric
video masking, key‑frame extraction and JPEG export with injected EXIF data.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:        # package not installed
    __version__ = "0.0.0"
