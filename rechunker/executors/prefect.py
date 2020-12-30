from typing import Iterable, Tuple

import prefect

from rechunker.types import Stage, ParallelPipelines, Executor


class PrefectExecutor(Executor[prefect.Flow]):
    """An execution engine based on Prefect.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects. Array must also be
    serializable by Prefect (i.e., with pickle).

    Execution plans for PrefectExecutor are prefect.Flow objects.
    """

    def prepare_plan(self, pipelines: ParallelPipelines) -> prefect.Flow:
        return _make_flow(pipelines)

    def execute_plan(self, plan: prefect.Flow, **kwargs):
        return plan.run(**kwargs)


class StageTaskWrapper(prefect.Task):
    def __init__(self, stage, **kwargs):
        self.stage = stage
        super().__init__(**kwargs)

    def run(self, key):
        return self.stage.func(key)


def _make_flow(pipelines: ParallelPipelines) -> prefect.Flow:
    with prefect.Flow("Rechunker") as flow:
        # iterate over different arrays in the group
        for pipeline in pipelines:
            stage_tasks = []
            # iterate over the different stages of the array copying
            for stage in pipeline:
                if stage.map_args is None:
                    stage_task = StageTaskWrapper(stage)()
                else:
                    stage_task= StageTaskWrapper(stage).map(stage.map_args)
                stage_tasks.append(stage_task)
            # create dependence between stages
            for n in range(len(stage_tasks) - 1):
                stage_tasks[n + 1].set_upstream(stage_tasks[n])
    return flow
