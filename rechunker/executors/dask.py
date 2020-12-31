from functools import reduce
from typing import Iterable, Tuple
import uuid
from pprint import pprint

import dask
import dask.array
from dask.delayed import Delayed
import dask.bag
from dask.optimization import fuse

from rechunker.types import Stage, MultiStagePipeline, ParallelPipelines, Executor


class DaskExecutor(Executor[Delayed]):
    """An execution engine based on dask.

    Supports zarr and dask arrays as inputs. Outputs must be zarr arrays.

    Execution plans for DaskExecutors are dask.delayed objects.
    """

    def prepare_plan(self, pipelines: ParallelPipelines) -> Delayed:
        return _make_pipelines(pipelines)

    def execute_plan(self, plan: Delayed, **kwargs):
        return plan.compute(**kwargs)


def _make_pipelines(pipelines: ParallelPipelines) -> Delayed:
    pipelines_delayed = [_make_pipeline(pipeline) for pipeline in pipelines]
    return _merge(*pipelines_delayed)


def _make_pipeline(pipeline: MultiStagePipeline) -> Delayed:
    stages_delayed = [_make_stage(stage) for stage in pipeline]
    return reduce(_add_upstream, stages_delayed)


def _make_stage(stage: Stage) -> Delayed:
    if stage.map_args is None:
        return dask.delayed(stage.func)()
    else:
        inputs = dask.bag.from_sequence(stage.map_args)
        bag = inputs.map(stage.func)
        return _add_barrier(bag)


def _merge_task(*args):
    pass


def _merge(*args: Iterable[Delayed]) -> Delayed:
    name = 'merge-' + dask.base.tokenize(*args)
    keys = [arg.key for arg in args]
    new_task = (_merge_task, *keys)
    graph = dask.base.merge(*[dask.utils.ensure_dict(d.dask) for d in args])
    graph[name] = new_task
    pprint(graph)
    return Delayed(name, graph)


def _barrier(*args):
    pass


def _add_barrier(collection):
    graph = dask.utils.ensure_dict(collection.dask)
    name = 'barrier-' + dask.base.tokenize(collection)
    new_task = (_barrier, *list(collection.__dask_keys__()))
    graph[name] = new_task
    return Delayed(name, graph)


def _add_upstream(first: Delayed, second: Delayed):
    upstream_key = first.key
    dsk = second.dask
    top_layer = _get_top_layer(dsk)
    new_top_layer = {}

    for key, value in top_layer.items():
        new_top_layer[key] = ((lambda a, b: a), value, upstream_key)

    dsk_new = dask.base.merge(
        dask.utils.ensure_dict(first.dask),
        dask.utils.ensure_dict(dsk),
        new_top_layer
    )

    return Delayed(second.key, dsk_new)


def _get_top_layer(dsk):
    if hasattr(dsk, 'layers'):
        # this is a HighLevelGraph
        top_layer_key = list(dsk.layers)[0]
        top_layer = dsk.layers[top_layer_key]
    else:
        first_key = next(iter(dsk))
        first_task = first_key[0].split('-')[0]
        top_layer = {k: v for k, v in dsk.items()
                     if k[0].startswith(first_task + '-')}
    return top_layer
