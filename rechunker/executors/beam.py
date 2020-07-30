import uuid
from typing import Iterable, NamedTuple, Optional, Mapping, Sequence, Tuple

import apache_beam as beam
import numpy as np

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
    # This would offer a cleaner API and would perhaps be faster, too.

    def prepare_plan(self, specs: Iterable[StagedCopySpec]) -> beam.PTransform:
        return "Rechunker" >> _OnDiskRechunker(specs)

    def execute_plan(self, plan: beam.PTransform, **kwargs):
        with beam.Pipeline(**kwargs) as pipeline:
            pipeline | plan


class _OnDiskRechunker(beam.PTransform):
    def __init__(self, specs: Iterable[StagedCopySpec]):
        super().__init__()
        self.specs = tuple(specs)

    def expand(self, pcoll):
        max_depth = max(len(spec.stages) for spec in self.specs)
        specs_map = {uuid.uuid1().hex: spec for spec in self.specs}

        # we explicitly thread target_id through each stage to ensure that they
        # are executed in order
        # TODO: consider refactoring to use Beam's ``Source`` API for improved
        # performance:
        # https://beam.apache.org/documentation/io/developing-io-overview/
        pcoll = pcoll | "Create" >> beam.Create(specs_map.keys())
        for stage in range(max_depth):
            specs_by_target = {
                k: v.stages[stage] if stage < len(v.stages) else None
                for k, v in specs_map.items()
            }
            pcoll = pcoll | f"Stage{stage}" >> _OnDiskStage(specs_by_target)
        return pcoll


class _OnDiskStage(beam.PTransform):
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


class _DirectCopySpec(NamedTuple):
    uuid: str
    source: ReadableArray
    target: WriteableArray
    read_chunks: Tuple[int, ...]
    intermediate_chunks: Tuple[int, ...]
    write_chunks: Tuple[int, ...]


class _DirectRechunker(beam.PTransform):
    def __init__(self, specs: Iterable[_DirectCopySpec]):
        super().__init__()
        self.specs = tuple(specs)

    def expand(self, pcoll):
        return (
            pcoll
            | "Create" >> beam.Create(self.specs)
            | "CreateTasks" >> beam.FlatMapTuple(_create_tasks)
            | "Reshuffle" >> beam.Reshuffle()
            | "ReadChunks" >> beam.Map(_read_chunk)
            | "SplitChunks" >> beam.FlatMap(_split_chunks)
            | "AddTargetIndex" >> beam.Map(_prepend_target_index)
            | "ConsolidateChunks" >> beam.CombinePerKey(_combine_chunks)
            | "WriteChunks" >> beam.Map(_write_chunk)
        )


def _create_tasks(spec):
    for key in chunk_keys(spec.source.shape, spec.read_chunks):
        yield spec, key


def _read_chunk(spec, key):
    return spec, key, spec.source[key]


def _split_chunks(spec, key, value):
    for k, v in _split_into_chunks(key, value, spec.intermediate_chunks):
        yield spec, k, v


def _prepend_target_index(spec, key, value):
    index = _chunk_index(key, spec.target_chunks)
    return (spec.uuid, index), (spec, key, value)


def _combine_chunks(triplets):
    identical_specs, keys, values = zip(*triplets)
    key, value = _conslidate_into_chunk(keys, values)
    return identical_specs[0], key, value


def _write_chunk(spec, key, value):
    spec.target[key] = value


def _chunk_index(key: Tuple[slice, ...], chunks: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(k.start // c for k, c in zip(key, chunks))


def _split_into_chunks(
    key: Tuple[slice, ...], value: ReadableArray, chunks: Tuple[int, ...],
) -> Tuple[Tuple[slice, ...], ReadableArray]:
    for key2 in chunk_keys(value.shape, chunks):
        fixed_key = tuple(
            slice(k1.start + k2.start, min(k1.start + k2.stop, k1.stop))
            for k1, k2 in zip(key, key2)
        )
        yield fixed_key, value[key2]


def _conslidate_into_chunk(
    keys: Sequence[Tuple[slice, ...]], values: Sequence[ReadableArray],
) -> Tuple[Tuple[slice, ...], ReadableArray]:
    lower = tuple(map(min, zip(*[[k.start for k in key] for key in keys])))
    upper = tuple(map(max, zip(*[[k.stop for k in key] for key in keys])))
    overall_key = tuple(map(slice, lower, upper))

    shape = tuple(u - l for l, u in zip(lower, upper))
    dtype = values[0].dtype
    assert all(dtype == v.dtype for v in values[1:])
    result = np.empty(shape, dtype)

    for key, value in zip(keys, values):
        fixed_key = tuple(slice(k.start - l, k.stop - l) for k, l in zip(key, lower))
        result[fixed_key] = value

    return overall_key, result
