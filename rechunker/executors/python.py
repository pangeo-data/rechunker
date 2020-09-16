from functools import partial

from typing import Callable, Iterable, Tuple

from rechunker.executors.util import chunk_keys, split_into_direct_copies
from rechunker.types import CopySpec, Executor, ReadableArray, WriteableArray


# PythonExecutor represents delayed execution tasks as functions that require
# no arguments.
Task = Callable[[], None]


class PythonExecutor(Executor[Task]):
    """An execution engine based on Python loops.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects.

    Execution plans for PythonExecutor are functions that accept no arguments.
    """

    def prepare_plan(self, specs: Iterable[CopySpec]) -> Task:
        tasks = []
        for spec in specs:
            for direct_spec in split_into_direct_copies(spec):
                tasks.append(partial(_direct_array_copy, *direct_spec))
        return partial(_execute_all, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        plan()


def _direct_array_copy(
    source: ReadableArray, target: WriteableArray, chunks: Tuple[int, ...]
) -> None:
    """Direct copy between arrays."""
    for key in chunk_keys(source.shape, chunks):
        target[key] = source[key]


def _execute_all(tasks: Iterable[Task]) -> None:
    for task in tasks:
        task()
