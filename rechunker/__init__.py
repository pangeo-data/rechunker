"""Top-level package for Zarr Rechunker."""
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .api import rechunk, Rechunked
