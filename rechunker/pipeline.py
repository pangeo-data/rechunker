import itertools
import math
from typing import Any, Iterable, Iterator, Tuple

import dask
import numpy as np

from .types import CopySpec, CopySpecExecutor, ParallelPipelines, Pipeline, Stage


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
    # typing won't work until we start using numpy types
    shape = spec.read.array.shape  # type: ignore
    if spec.intermediate.array is None:
        stages = [
            Stage(
                copy_read_to_write,
                "copy_read_to_write",
                mappable=chunk_keys(shape, spec.write.chunks),
            )
        ]
    else:
        stages = [
            Stage(
                copy_read_to_intermediate,
                "copy_read_to_intermediate",
                mappable=chunk_keys(shape, spec.intermediate.chunks),
            ),
            Stage(
                copy_intermediate_to_write,
                "copy_intermediate_to_write",
                mappable=chunk_keys(shape, spec.write.chunks),
            ),
        ]
    return Pipeline(stages, config=spec)


def specs_to_pipelines(specs: Iterable[CopySpec]) -> ParallelPipelines:
    return tuple((spec_to_pipeline(spec) for spec in specs))


class CopySpecToPipelinesMixin(CopySpecExecutor):
    def prepare_plan(self, specs: Iterable[CopySpec]):
        pipelines = specs_to_pipelines(specs)
        return self.pipelines_to_plan(pipelines)

    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> Any:
        """Transform ParallelPiplines to an execution plan"""
        raise NotImplementedError
