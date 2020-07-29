import uuid
from typing import Iterable, Optional, Mapping, Tuple

import apache_beam as beam

from rechunker.executors.util import chunk_keys
from rechunker.types import (
    CopySpec,
    StagedCopySpec,
    Executor,
    ReadableArray,
    WriteableArray,
)


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

        # we explicitly thread target_id through each stage to ensure that they
        # are executed in order
        pcoll = pcoll | "Create" >> beam.Create(specs_map.keys())
        for stage in range(max_depth):
            specs_by_target = {
                k: v.stages[stage] if stage < len(v.stages) else None
                for k, v in specs_map.items()
            }
            pcoll = pcoll | f"Stage{stage}" >> _CopyStage(specs_by_target)
        return pcoll


class _CopyStage(beam.PTransform):
    def __init__(self, specs_by_target: Mapping[str, CopySpec]):
        super().__init__()
        self.specs_by_target = specs_by_target

    def expand(self, pcoll):
        return (
            pcoll
            | "Start" >> beam.FlatMap(_start_stage, self.specs_by_target)
            | "CreateTasks" >> beam.FlatMapTuple(_copy_tasks)
            # prevent undesirable fusion
            # https://stackoverflow.com/a/54131856/809705
            | "Reshuffle" >> beam.Reshuffle()
            | "CopyChunks" >> beam.MapTuple(_copy_chunk)
            # prepare inputs for the next stage (if any)
            | "Finish" >> beam.Distinct()
        )


def _start_stage(
    target_id: str, specs_by_target: Mapping[str, Optional[CopySpec]],
) -> Tuple[str, CopySpec]:
    spec = specs_by_target[target_id]
    if spec is not None:
        yield target_id, spec


def _copy_tasks(
    target_id: str, spec: CopySpec
) -> Tuple[str, Tuple[slice, ...], ReadableArray, WriteableArray]:
    for key in chunk_keys(spec.source.shape, spec.chunks):
        yield target_id, key, spec.source, spec.target


def _copy_chunk(
    target_id: str,
    key: Tuple[slice, ...],
    source: ReadableArray,
    target: WriteableArray,
) -> str:
    target[key] = source[key]
    return target_id
