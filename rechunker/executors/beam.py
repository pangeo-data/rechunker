import uuid
from typing import Iterable, Iterator, Mapping, Tuple

import apache_beam as beam

from rechunker.executors.util import (
    DirectCopySpec,
    chunk_keys,
    split_into_direct_copies,
)
from rechunker.types import CopySpec, Executor, ReadableArray, WriteableArray


class BeamExecutor(Executor[beam.PTransform]):
    """An execution engine based on Apache Beam.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects. Array must also be
    serializable by Beam (i.e., with pickle).

    Execution plans for BeamExecutor are beam.PTransform objects.
    """

    # TODO: explore adding an option to do rechunking with Beam groupby
    # operations instead of explicitly writing intermediate arrays to disk.
    # This would offer a cleaner API and would perhaps be faster, too.

    def prepare_plan(self, specs: Iterable[CopySpec]) -> beam.PTransform:
        return "Rechunker" >> _Rechunker(specs)

    def execute_plan(self, plan: beam.PTransform, **kwargs):
        with beam.Pipeline(**kwargs) as pipeline:
            pipeline | plan


class _Rechunker(beam.PTransform):
    def __init__(self, specs: Iterable[CopySpec]):
        super().__init__()
        self.direct_specs = tuple(map(split_into_direct_copies, specs))

    def expand(self, pcoll):
        max_depth = max(len(copies) for copies in self.direct_specs)
        specs_map = {uuid.uuid1().hex: copies for copies in self.direct_specs}

        # we explicitly thread target_id through each stage to ensure that they
        # are executed in order
        # TODO: consider refactoring to use Beam's ``Source`` API for improved
        # performance:
        # https://beam.apache.org/documentation/io/developing-io-overview/
        pcoll = pcoll | "Create" >> beam.Create(specs_map.keys())
        for stage in range(max_depth):
            specs_by_target = {
                k: v[stage] if stage < len(v) else None for k, v in specs_map.items()
            }
            pcoll = pcoll | f"Stage{stage}" >> _CopyStage(specs_by_target)
        return pcoll


class _CopyStage(beam.PTransform):
    def __init__(self, specs_by_target: Mapping[str, DirectCopySpec]):
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
    target_id: str, specs_by_target: Mapping[str, DirectCopySpec],
) -> Iterator[Tuple[str, DirectCopySpec]]:
    spec = specs_by_target.get(target_id)
    if spec is not None:
        yield target_id, spec


def _copy_tasks(
    target_id: str, spec: DirectCopySpec
) -> Iterator[Tuple[str, Tuple[slice, ...], ReadableArray, WriteableArray]]:
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
