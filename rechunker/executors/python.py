from functools import partial
from typing import Callable, Iterable

from rechunker.executors.pipeline import CopySpecToPipelinesMixin
from rechunker.types import ParallelPipelines, PipelineExecutor, Stage

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
                tasks.append(partial(_execute_stage, stage))
        return partial(_execute_all, tasks)

    def execute_plan(self, plan: Task, **kwargs):
        plan()


class PythonCopySpecExecutor(PythonPipelineExecutor, CopySpecToPipelinesMixin):
    pass


def _execute_stage(stage: Stage) -> Task:
    if stage.map_args is None:
        return stage.func
    for f in map(stage.func, stage.map_args):
        pass


def _execute_all(tasks: Iterable[Task]) -> None:
    for task in tasks:
        task()
