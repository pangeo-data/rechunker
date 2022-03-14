import itertools
import math
from typing import Any, Iterable, Iterator, Tuple, TypeVar

import dask
import numpy as np

from .types import (
    CopySpec,
    CopySpecExecutor,
    Stage,
    Pipeline,
    ParallelPipelines,
    ReadableArray,
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
        with dask.config.set(scheduler="single-threaded"):
            data = np.asarray(source[chunk_key])
        target[chunk_key] = data

    keys = list(chunk_keys(source.shape, chunks))
    return Stage(_copy_chunk, keys)


def copy_read_to_write(chunk_key, *, config=CopySpec):
    with dask.config.set(scheduler="single-threaded"):
        data = np.asarray(config.read.array[chunk_key])
    config.write.array[chunk_key] = data


def copy_read_to_intermediate(chunk_key, *, config=CopySpec):
    with dask.config.set(scheduler="single-threaded"):
        data = np.asarray(config.read.array[chunk_key])
    config.intermediate.array[chunk_key] = data


def copy_intermediate_to_write(chunk_key, *, config=CopySpec):
    with dask.config.set(scheduler="single-threaded"):
        data = np.asarray(config.intermediate.array[chunk_key])
    config.write.array[chunk_key] = data


def spec_to_pipeline(spec: CopySpec) -> Pipeline:
    if spec.intermediate.array is None:
        stages = [
            Stage(
                copy_read_to_write,
                "copy_read_to_write",
                mappable=chunk_keys(spec.read.array.shape, spec.read.chunks),
            )
        ]
    else:
        stages = [
            Stage(
                copy_read_to_intermediate,
                "copy_read_to_intermediate",
                mappable=chunk_keys(spec.read.array.shape, spec.read.chunks),
            ),
            Stage(
                copy_intermediate_to_write,
                "copy_intermediate_to_write",
                mappable=chunk_keys(spec.intermediate.array.shape, spec.intermediate.chunks),
            )
        ]
    return Pipeline(stages, config=spec)


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
