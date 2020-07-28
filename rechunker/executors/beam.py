from functools import partial
import itertools
import math
import uuid
from typing import Any, Iterable, Optional, Mapping, Tuple

import apache_beam as beam

from rechunker.types import CopySpec, StagedCopySpec, Executor


class BeamExecutor(Executor[beam.PTransform]):
    """An execution engine based on Apache Beam.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects. Array must also be
    serializable by Beam (i.e., with pickle).

    Execution plans for BeamExecutor are beam.PTransform objects.
    """

    # TODO: explore adding an option to do rechunking with Beam groupby
    # operations instead of explicitly writing intermediate arrays to disk.

    def prepare_plan(self, specs: Iterable[StagedCopySpec]) -> beam.PTransform:
        return "Rechunker" >> _Rechunker(specs)

    def execute_plan(self, plan: beam.PTransform, **kwargs):
        with beam.Pipeline(**kwargs) as pipeline:
            pipeline | plan


class _Rechunker(beam.PTransform):
    def __init__(self, specs: Iterable[StagedCopySpec]):
        super().__init__()
        self.specs = tuple(specs)

    def expand(self, pcoll):
        max_depth = max(len(spec.stages) for spec in self.specs)
        specs_map = {uuid.uuid1().hex: spec for spec in self.specs}

        pcoll = pcoll | "Create" >> beam.Create([(k, None) for k in specs_map])
        for stage in range(max_depth):
            specs_by_target = {
                k: v.stages[stage] if stage < len(v.stages) else None
                for k, v in specs_map.items()
            }
            pcoll = pcoll | f"Stage{stage}" >> _Stage(specs_by_target)
        return pcoll


class _Stage(beam.PTransform):
    def __init__(self, specs_by_target: Mapping[str, CopySpec]):
        super().__init__()
        self.specs_by_target = specs_by_target

    def expand(self, pcoll):
        start_fn = partial(_start_stage, self.specs_by_target)
        return (
            pcoll
            | "Start" >> beam.FlatMapTuple(start_fn)
            | "CreateTasks" >> beam.FlatMapTuple(_copy_tasks)
            | "Reshuffle" >> beam.Reshuffle()  # prevent undesirable fusion
            | "CopyChunks" >> beam.MapTuple(_copy_chunk)
            | "Finish" >> beam.GroupByKey()
        )


def _start_stage(
    specs_by_target_id: Mapping[str, Optional[CopySpec]],
    target_id: str,
    unused_value: Any,
) -> Tuple[str, CopySpec]:
    spec = specs_by_target_id[target_id]
    if spec is not None:
        yield target_id, spec


def _copy_tasks(target_id: str, spec: CopySpec):
    for key in _chunked_keys(spec.source.shape, spec.chunks):
        yield target_id, key, spec.source, spec.target


def _chunked_keys(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Tuple[slice, ...]:
    ranges = [range(math.ceil(s / c)) for s, c in zip(shape, chunks)]
    for indices in itertools.product(*ranges):
        yield tuple(slice(c * i, c * (i + 1)) for i, c in zip(indices, chunks))


def _copy_chunk(target_id, key, source, target):
    target[key] = source[key]
    return (target_id, None)
