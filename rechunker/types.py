"""Types definitions used by executors."""
from typing import Any, Callable, Generic, Iterable, NamedTuple, Optional, Tuple, TypeVar

# TODO: replace with Protocols, once Python 3.8+ is required
Array = Any
ReadableArray = Any
WriteableArray = Any


class ArrayProxy(NamedTuple):
    """Representation of a chunked array for reads and writes.

    Attributes
    ----------
    array : array or None
        If ``array`` is None, it represents an array without any on-disk
        representation. Otherwise ``array`` should be an explicit array to read
        and/or write to.
    chunks : tuple
        Chunks to use when reading/writing from this array. If ``array`` is
        chunked, ``array.chunks`` will always even divide ``chunks``.
    """

    array: Optional[Array]
    chunks: Tuple[int, ...]


class CopySpec(NamedTuple):
    """Specification for how to rechunk an array using a single intermediate.

    Attributes
    ----------
    read : ArrayProxy
        Read proxy with an ``array`` attribute that supports ``__getitem__``.
    intermediate : ArrayProxy
        Intermediate proxy with either an ``array`` that is either ``None``
        (no intermediate storage on disk) or that supports both ``__getitem__``
        and ``__setitem__``. The ``chunks`` on intermediates are technically
        redundant (they the elementwise minimum of the read and write chunks)
        but they are provided for convenience.
    write : ArrayProxy
        Write proxy with an ``array`` attribute that supports ``__setitem__``.
    """

    read: ArrayProxy
    intermediate: ArrayProxy
    write: ArrayProxy


class Stage(NamedTuple):
    """A single stage of a pipeline.

    Attributes
    ----------
    func: Callable
        A function to be called in this stage. Accepts either 0 or 1 arguments.
    map_args: List, Optional
        Arguments which will be mapped to the function
    annotations: Dict, Optional
        Hints provided to the scheduler
    """

    func: Callable
    map_args: Optional[Iterable]
    # TODO: figure out how to make optional
    # annotations: Dict


MultiStagePipeline = Iterable[Stage]
ParallelPipelines = Iterable[MultiStagePipeline]

T = TypeVar("T")


class Executor(Generic[T]):
    """Base class for execution engines.

    Executors prepare and execute scheduling plans, in whatever form is most
    convenient for users of that executor.
    """

    # TODO: add support for multi-stage copying plans (in the form of a new,
    # dedicated method)

    def prepare_plan(self, pipeline: ParallelPipelines) -> T:
        """Convert copy specifications into a plan."""
        raise NotImplementedError

    def execute_plan(self, plan: T, **kwargs):
        """Execute a plan."""
        raise NotImplementedError
