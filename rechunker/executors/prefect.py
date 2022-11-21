from typing import List

from prefect import Flow, task, unmapped

from rechunker.types import ParallelPipelines, PipelineExecutor


class PrefectPipelineExecutor(PipelineExecutor[Flow]):
    """An execution engine based on Prefect.

    Execution plans for PrefectExecutor are prefect.Flow objects.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Flow:
        with Flow("rechunker") as flow:
            for pipeline in pipelines:
                upstream_tasks = []  # type: List[task]
                for stage in pipeline.stages:
                    stage_task = task(stage.function, name=stage.name)
                    if stage.mappable is not None:
                        stage_task_called = stage_task.map(
                            list(stage.mappable),  # prefect doesn't accept a generator
                            config=unmapped(pipeline.config),
                            upstream_tasks=[unmapped(t) for t in upstream_tasks],
                        )
                    else:
                        stage_task_called = stage_task(
                            config=pipeline.config, upstream_tasks=upstream_tasks
                        )
                    upstream_tasks = [stage_task_called]
        return flow

    def execute_plan(self, plan: Flow, **kwargs):
        state = plan.run(**kwargs)
        return state
