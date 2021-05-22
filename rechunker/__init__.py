"""Top-level package for Zarr Rechunker."""
try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "unknown"

from .algorithm import rechunking_plan
from .api import Rechunked, rechunk
