from __future__ import annotations

from typing import Any, Dict, Set, Tuple, Union

import dask
from dask.blockwise import BlockwiseDepDict, blockwise
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from packaging.version import Version

from rechunker.types import ParallelPipelines, Pipeline, PipelineExecutor

# Change in how dask collection token are given to blockwise()
if Version(dask.__version__) >= Version("2025.1.0"):
    # Public module exposed in 2025.1
    from dask.task_spec import TaskRef
elif Version(dask.__version__) >= Version("2024.12.0"):
    # TaskRef introduced in dask 2024.9, but only necessary in block wise as of 2024.12
    from dask._task_spec import TaskRef
else:
    TaskRef = lambda x: x


def wrap_map_task(function):
    # dependencies are dummy args used to create dependence between stages
    def wrapped(map_arg, config, *dependencies):
        return function(map_arg, config=config)

    return wrapped


def wrap_standalone_task(function):
    def wrapped(config, *dependencies):
        return function(config=config)

    return wrapped


def checkpoint(*args):
    return


def append_token(task_name: str, token: str) -> str:
    return f"{task_name}-{token}"


class DaskPipelineExecutor(PipelineExecutor[Delayed]):
    """An execution engine based on dask.

    Supports zarr and dask arrays as inputs. Outputs must be zarr arrays.

    Execution plans for DaskExecutors are dask.delayed objects.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Delayed:
        return [_make_pipeline(pipeline) for pipeline in pipelines]

    def execute_plan(self, plan: Delayed, **kwargs):
        return dask.compute(*plan, **kwargs)


def _make_pipeline(pipeline: Pipeline) -> Delayed:
    token = dask.base.tokenize(pipeline)

    # we are constructing a HighLevelGraph from scratch
    # https://docs.dask.org/en/latest/high-level-graphs.html
    layers = dict()  # type: Dict[str, Dict[Union[str, Tuple[str, int]], Any]]
    dependencies = dict()  # type: Dict[str, Set[str]]

    # start with just the config as a standalone layer
    # create a custom delayed object for the config
    config_key = append_token("config", token)
    layers[config_key] = {config_key: pipeline.config}
    dependencies[config_key] = set()

    prev_key: str = config_key
    for stage in pipeline.stages:
        if stage.mappable is None:
            stage_key = append_token(stage.name, token)
            func = wrap_standalone_task(stage.function)
            layers[stage_key] = {stage_key: (func, config_key, prev_key)}
            dependencies[stage_key] = {config_key, prev_key}
        else:
            func = wrap_map_task(stage.function)
            map_key = append_token(stage.name, token)
            layers[map_key] = map_layer = blockwise(
                func,
                map_key,
                "x",  # <-- dimension name doesn't matter
                BlockwiseDepDict({(i,): x for i, x in enumerate(stage.mappable)}),
                # ^ this is extra annoying. `BlockwiseDepList` at least would be nice.
                "x",
                TaskRef(config_key),
                None,
                TaskRef(prev_key),
                None,
                numblocks={},
                # ^ also annoying; the default of None breaks Blockwise
            )
            dependencies[map_key] = {config_key, prev_key}

            stage_key = f"{stage.name}-checkpoint-{token}"
            layers[stage_key] = {stage_key: (checkpoint, *map_layer.get_output_keys())}
            dependencies[stage_key] = {map_key}
        prev_key = stage_key

    hlg = HighLevelGraph(layers, dependencies)
    delayed = Delayed(prev_key, hlg)
    return delayed
