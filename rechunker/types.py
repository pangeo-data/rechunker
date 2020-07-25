"""Types definitions used by executors."""
from typing import Any, Generic, Iterable, NamedTuple, Tuple, TypeVar

import zarr


class CopySpec(NamedTuple):
    """Specifcation of how to copy between two arrays."""

    # TODO: remove Any by making CopySpec a Generic, once we only support Python
    # 3.7+: https://stackoverflow.com/questions/50530959
    source: Any
    target: zarr.Array
    chunks: Tuple[int, ...]


class StagedCopySpec:
    stages: Tuple[CopySpec, ...]

    def __init__(self, stages: Iterable[CopySpec]):
        self.stages = tuple(stages)


T = TypeVar("T")


class Executor(Generic[T]):
    """Base class for execution engines.

    Executors prepare and execute scheduling plans, in whatever form is most
    convenient for users of that executor.
    """

    def prepare_plan(self, specs: Iterable[StagedCopySpec]) -> T:
        """Convert copy specifications into a plan."""
        raise NotImplementedError

    def execute_plan(self, plan: T, **kwargs):
        """Execute a plan."""
        raise NotImplementedError
