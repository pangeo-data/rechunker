from functools import partial
from typing import Callable, Iterable

from rechunker.types import ParallelPipelines, PipelineExecutor

# PythonExecutor represents delayed execution tasks as functions that require
# no arguments.
Task = Callable[[], None]


class PythonPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Python loops.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects.

    Execution plans for PythonExecutor are functions that accept no arguments.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Task:
        tasks = []
        for pipeline in pipelines:
            for stage in pipeline:
                if stage.map_args is None:
                    tasks.append(stage.func)
                else:
                    for arg in stage.map_args:
                        tasks.append(partial(stage.func, arg))
        return partial(_execute_all, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        plan()


def _execute_all(tasks: Iterable[Task]) -> None:
    for task in tasks:
        task()
