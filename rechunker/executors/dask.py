from functools import reduce
from typing import Iterable

import dask
import dask.array
from dask.delayed import Delayed

from rechunker.types import (
    MultiStagePipeline,
    ParallelPipelines,
    PipelineExecutor,
    Stage,
)


class DaskPipelineExecutor(PipelineExecutor[Delayed]):
    """An execution engine based on dask.

    Supports zarr and dask arrays as inputs. Outputs must be zarr arrays.

    Execution plans for DaskExecutors are dask.delayed objects.
    """

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Delayed:
        return _make_pipelines(pipelines)

    def execute_plan(self, plan: Delayed, **kwargs):
        return plan.compute(**kwargs)


def _make_pipelines(pipelines: ParallelPipelines) -> Delayed:
    pipelines_delayed = [_make_pipeline(pipeline) for pipeline in pipelines]
    return _merge(*pipelines_delayed)


def _make_pipeline(pipeline: MultiStagePipeline) -> Delayed:
    stages_delayed = [_make_stage(stage) for stage in pipeline]
    d = reduce(_add_upstream, stages_delayed)
    return d


def _make_stage(stage: Stage) -> Delayed:
    if stage.map_args is None:
        return dask.delayed(stage.func)()
    else:
        name = stage.func.__name__ + "-" + dask.base.tokenize(stage.func)
        dsk = {(name, i): (stage.func, arg) for i, arg in enumerate(stage.map_args)}
        # create a barrier
        top_key = "stage-" + dask.base.tokenize(stage.func, stage.map_args)

        def merge_all(*args):
            # this function is dependent on its arguments but doesn't actually do anything
            return None

        dsk.update({top_key: (merge_all, *list(dsk))})
        return Delayed(top_key, dsk)


def _merge_task(*args):
    pass


def _merge(*args: Iterable[Delayed]) -> Delayed:
    name = "merge-" + dask.base.tokenize(*args)
    # mypy doesn't like arg.key
    keys = [getattr(arg, "key") for arg in args]
    new_task = (_merge_task, *keys)
    # mypy doesn't like arg.dask
    graph = dask.base.merge(
        *[dask.utils.ensure_dict(getattr(arg, "dask")) for arg in args]
    )
    graph[name] = new_task
    d = Delayed(name, graph)
    return d


def _add_upstream(first: Delayed, second: Delayed):
    upstream_key = first.key
    dsk = second.dask
    top_layer = _get_top_layer(dsk)
    new_top_layer = {}

    for key, value in top_layer.items():
        new_top_layer[key] = ((lambda a, b: a), value, upstream_key)

    dsk_new = dask.base.merge(
        dask.utils.ensure_dict(first.dask), dask.utils.ensure_dict(dsk), new_top_layer
    )

    return Delayed(second.key, dsk_new)


def _get_top_layer(dsk):
    if hasattr(dsk, "layers"):
        # this is a HighLevelGraph
        top_layer_key = list(dsk.layers)[0]
        top_layer = dsk.layers[top_layer_key]
    else:
        # could this go wrong?
        first_key = next(iter(dsk))
        first_task = first_key[0].split("-")[0]
        top_layer = {k: v for k, v in dsk.items() if k[0].startswith(first_task + "-")}
    return top_layer
