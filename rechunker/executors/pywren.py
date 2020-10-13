from functools import partial

from typing import Callable, Iterable, Tuple

from rechunker.executors.util import chunk_keys, split_into_direct_copies
from rechunker.types import CopySpec, Executor, ReadableArray, WriteableArray

import pywren_ibm_cloud as pywren
from pywren_ibm_cloud.executor import FunctionExecutor

# PywrenExecutor represents delayed execution tasks as functions that require
# a FunctionExecutor.
Task = Callable[[FunctionExecutor], None]


class PywrenExecutor(Executor[Task]):
    """An execution engine based on Pywren.

    Supports zarr arrays as inputs. Outputs must be zarr arrays.

    Any Pywren FunctionExecutor can be passed to the constructor. By default
    a Pywren `local_executor` will be used

    Execution plans for PywrenExecutor are functions that accept no arguments.
    """

    def __init__(self, pywren_function_executor: FunctionExecutor = None):
        self.pywren_function_executor = pywren_function_executor

    def prepare_plan(self, specs: Iterable[CopySpec]) -> Task:
        tasks = []
        for spec in specs:
            # Tasks for a single spec must be executed in series
            spec_tasks = []
            for direct_spec in split_into_direct_copies(spec):
                spec_tasks.append(partial(_direct_array_copy, *direct_spec))
            tasks.append(partial(_execute_in_series, spec_tasks))
        # TODO: execute tasks for different specs in parallel
        return partial(_execute_in_series, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        if self.pywren_function_executor is None:
            # No Pywren function executor specified, so use a local one, and shutdown after use
            with pywren_local_function_executor() as pywren_function_executor:
                plan(pywren_function_executor)
        else:
            plan(self.pywren_function_executor)


def pywren_local_function_executor():
    return pywren.local_executor(
        # Minimal config needed to avoid Pywren error if ~/.pywren_config is missing
        config={"pywren": {"storage_bucket": "unused"}}
    )


def _direct_array_copy(
    source: ReadableArray,
    target: WriteableArray,
    chunks: Tuple[int, ...],
    pywren_function_executor: FunctionExecutor,
) -> None:
    """Direct copy between arrays using Pywren for parallelism"""
    iterdata = [(source, target, key) for key in chunk_keys(source.shape, chunks)]

    def direct_copy(iterdata):
        source, target, key = iterdata
        target[key] = source[key]

    futures = pywren_function_executor.map(direct_copy, iterdata)
    pywren_function_executor.get_result(futures)


def _execute_in_series(
    tasks: Iterable[Task], pywren_function_executor: FunctionExecutor
) -> None:
    for task in tasks:
        task(pywren_function_executor)
