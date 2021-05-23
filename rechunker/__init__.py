"""Top-level package for Zarr Rechunker."""
try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "unknown"

from .algorithm import multistage_rechunking_plan, rechunking_plan
from .api import Rechunked, rechunk
