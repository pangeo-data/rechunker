import itertools
import math

from typing import Iterator, NamedTuple, Tuple

from rechunker.types import CopySpec, ReadableArray, WriteableArray


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


class DirectCopySpec(NamedTuple):
    """Specification of how to directly copy between two arrays."""

    source: ReadableArray
    target: WriteableArray
    chunks: Tuple[int, ...]


def split_into_direct_copies(spec: CopySpec) -> Tuple[DirectCopySpec, ...]:
    """Convert a rechunked copy into a list of direct copies."""
    if spec.intermediate.array is None:
        return (DirectCopySpec(spec.read.array, spec.write.array, spec.read.chunks,),)
    else:
        return (
            DirectCopySpec(spec.read.array, spec.intermediate.array, spec.read.chunks,),
            DirectCopySpec(
                spec.intermediate.array, spec.write.array, spec.write.chunks,
            ),
        )
