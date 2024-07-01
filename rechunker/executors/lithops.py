from typing import Any, Callable

import lithops

from rechunker.types import ParallelPipelines, PipelineExecutor

Task = Callable[[Any], None]


class LithopsPipelineExecutor(PipelineExecutor[Task]):
    """An execution engine based on Lithops framework."""

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Task:
        def map_function(stage, function, config):
            function(stage, config=config) if stage else function(config=config)

        def _prepare_input_data():
            iterdata = []
            for pipeline in pipelines:
                for stage in pipeline.stages:
                    if stage.mappable is not None:
                        for m in stage.mappable:
                            iterdata.append((m, stage.function, pipeline.config))
                    else:
                        iterdata.append((None, stage.function, pipeline.config))
            return iterdata

        def plan(config):
            iterdata = _prepare_input_data()

            if config is not None:
                fexec = lithops.FunctionExecutor(config=config)
            else:
                fexec = lithops.FunctionExecutor()

            fexec.map(map_function, iterdata[0])
            fexec.get_result()

        return plan

    def execute_plan(self, plan: Task, config=None, **kwargs):
        plan(config)
