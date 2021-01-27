import itertools
import math
from typing import Any, Iterable, Iterator, Tuple, TypeVar

import dask
import numpy as np

from .executors.dask import DaskPipelineExecutor
from .executors.prefect import PrefectPipelineExecutor
from .executors.python import PythonPipelineExecutor
from .types import (
    CopySpec,
    CopySpecExecutor,
    MultiStagePipeline,
    ParallelPipelines,
    ReadableArray,
    Stage,
    WriteableArray,
)


def chunk_keys(
    shape: Tuple[int, ...], chunks: Tuple[int, ...]
) -> Iterator[Tuple[slice, ...]]:
    """Iterator over array indexing keys of the desired chunk sized.

    The union of all keys indexes every element of an array of shape ``shape``
    exactly once. Each array resulting from indexing is of shape ``chunks``,
    except possibly for the last arrays along each dimension (if ``chunks``
    do not even divide ``shape``).
    """
    ranges = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
    for indices in itertools.product(*ranges):
        yield tuple(
            slice(c * i, min(c * (i + 1), s)) for i, s, c in zip(indices, shape, chunks)
        )


def copy_stage(
    source: ReadableArray, target: WriteableArray, chunks: Tuple[int, ...]
) -> Stage:
    # use a closure to eliminate extra arguments
    def _copy_chunk(chunk_key):
        # calling np.asarray here allows the source to be a dask array
        # TODO: could we asyncify this to operate in a streaming fashion
        # make sure this is not happening inside a dask scheduler
        print(f"_copy_chunk({chunk_key})")
        with dask.config.set(scheduler="single-threaded"):
            data = np.asarray(source[chunk_key])
        target[chunk_key] = data

    keys = list(chunk_keys(source.shape, chunks))
    return Stage(_copy_chunk, keys)


def spec_to_pipeline(spec: CopySpec) -> MultiStagePipeline:
    pipeline = []
    if spec.intermediate.array is None:
        pipeline.append(copy_stage(spec.read.array, spec.write.array, spec.read.chunks))
    else:
        pipeline.append(
            copy_stage(spec.read.array, spec.intermediate.array, spec.read.chunks)
        )
        pipeline.append(
            copy_stage(spec.intermediate.array, spec.write.array, spec.write.chunks)
        )
    return pipeline


def specs_to_pipelines(specs: Iterable[CopySpec]) -> ParallelPipelines:
    return [spec_to_pipeline(spec) for spec in specs]


T = TypeVar("T")


class CopySpecToPipelinesMixin(CopySpecExecutor):
    def prepare_plan(self, specs: Iterable[CopySpec]) -> T:
        pipelines = specs_to_pipelines(specs)
        return self.pipelines_to_plan(pipelines)

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Any:
        """Transform ParallelPiplines to an execution plan"""
        raise NotImplementedError


class PythonCopySpecExecutor(PythonPipelineExecutor, CopySpecToPipelinesMixin):
    pass


class DaskCopySpecExecutor(DaskPipelineExecutor, CopySpecToPipelinesMixin):
    pass


class PrefectCopySpecExecutor(PrefectPipelineExecutor, CopySpecToPipelinesMixin):
    pass
