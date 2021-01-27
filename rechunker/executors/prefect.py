import prefect

from rechunker.types import ParallelPipelines, PipelineExecutor


class PrefectPipelineExecutor(PipelineExecutor[prefect.Flow]):
    """An execution engine based on Prefect.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects. Array must also be
    serializable by Prefect (i.e., with pickle).

    Execution plans for PrefectExecutor are prefect.Flow objects.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> prefect.Flow:
        return _make_flow(pipelines)

    def execute_plan(self, plan: prefect.Flow, **kwargs):
        state = plan.run(**kwargs)
        return state


class MappedTaskWrapper(prefect.Task):
    def __init__(self, stage, **kwargs):
        self.stage = stage
        super().__init__(**kwargs)

    def run(self, key):
        return self.stage.func(key)


class SingleTaskWrapper(prefect.Task):
    def __init__(self, stage, **kwargs):
        self.stage = stage
        super().__init__(**kwargs)

    def run(self):
        return self.stage.func()


def _make_flow(pipelines: ParallelPipelines) -> prefect.Flow:
    with prefect.Flow("Rechunker") as flow:
        # iterate over different arrays in the group
        for pipeline in pipelines:
            stage_tasks = []
            # iterate over the different stages of the array copying
            for stage in pipeline:
                if stage.map_args is None:
                    stage_task = SingleTaskWrapper(stage)
                else:
                    stage_task = MappedTaskWrapper(stage).map(stage.map_args)
                stage_tasks.append(stage_task)
            # create dependence between stages
            for n in range(len(stage_tasks) - 1):
                stage_tasks[n + 1].set_upstream(stage_tasks[n])
    return flow
