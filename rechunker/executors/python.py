import itertools
from functools import partial
import math

from typing import Callable, Iterable

from rechunker.types import CopySpec, StagedCopySpec, Executor


# PythonExecutor represents delayed execution tasks as functions that require
# no arguments.
Task = Callable[[], None]


class PythonExecutor(Executor[Task]):
    """An execution engine based on Python loops.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects.

    Execution plans for PythonExecutor are functions that accept no arguments.
    """

    def prepare_plan(self, specs: Iterable[StagedCopySpec]) -> Task:
        tasks = []
        for staged_copy_spec in specs:
            for copy_spec in staged_copy_spec.stages:
                tasks.append(partial(_direct_copy_array, copy_spec))
        return partial(_execute_all, tasks)

    def execute_plan(self, plan: Task):
        plan()


def _direct_copy_array(copy_spec: CopySpec) -> None:
    """Direct copy between zarr arrays."""
    source_array, target_array, chunks = copy_spec
    shape = source_array.shape
    ranges = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
    for indices in itertools.product(*ranges):
        key = tuple(slice(c * i, c * (i + 1)) for i, c in zip(indices, chunks))
        target_array[key] = source_array[key]


def _execute_all(tasks: Iterable[Task]) -> None:
    for task in tasks:
        task()
