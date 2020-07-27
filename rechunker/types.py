"""Types definitions used by executors."""
from typing import Any, Generic, Iterable, NamedTuple, Tuple, TypeVar


ReadableArray = Any
WriteableArray = Any


class CopySpec(NamedTuple):
    """Specifcation of how to copy between two arrays."""

    # TODO: remove Any by making CopySpec a Generic, once we only support Python
    # 3.7+: https://stackoverflow.com/questions/50530959
    source: ReadableArray
    target: WriteableArray
    chunks: Tuple[int, ...]


class StagedCopySpec:
    """Specification of a copying process involving intermediate arrays.

    The stages in a staged copy process must be completed in order. The
    ``target`` of each stage corresponds to the ``source`` of the following
    stage.
    """

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
