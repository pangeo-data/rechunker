"""Top-level package for Zarr Rechunker."""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .api import rechunk, Rechunked
