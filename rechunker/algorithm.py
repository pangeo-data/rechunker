"""Core rechunking algorithm stuff."""
from math import floor
from typing import List, Optional, Sequence, Tuple

from rechunker.compat import prod


def consolidate_chunks(
    shape: Sequence[int],
    chunks: Sequence[int],
    itemsize: int,
    max_mem: int,
    chunk_limits: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[int, ...]:
    """
    Consolidate input chunks up to a certain memory limit. Consolidation starts on the
    highest axis and proceeds towards axis 0.

    Parameters
    ----------
    shape : Tuple
        Array shape
    chunks : Tuple
        Original chunk shape (must be in form (5, 10, 20), no irregular chunks)
    max_mem : Int
        Maximum permissible chunk memory size, measured in units of itemsize
    chunk_limits : Tuple, optional
        Maximum size of each chunk along each axis. If None, don't consolidate
        axis. If -1, no limit.

    Returns
    -------
    new_chunks : tuple
        The new chunks, size guaranteed to be <= mam_mem
    """

    ndim = len(shape)
    if chunk_limits is None:
        chunk_limits = shape
    assert len(chunk_limits) == ndim

    # now convert chunk_limits to a dictionary
    # key: axis, value: limit
    chunk_limit_per_axis = {}
    for n_ax, cl in enumerate(chunk_limits):
        if cl is not None:
            if cl == -1:
                chunk_limit_per_axis[n_ax] = shape[n_ax]
            elif chunks[n_ax] <= cl <= shape[n_ax]:
                chunk_limit_per_axis[n_ax] = cl
            elif cl > shape[n_ax]:
                chunk_limit_per_axis[n_ax] = shape[n_ax]
            else:
                raise ValueError(f"Invalid chunk_limits {chunk_limits}.")

    chunk_mem = itemsize * prod(chunks)
    if chunk_mem > max_mem:
        raise ValueError(f"chunk_mem {chunk_mem} > max_mem {max_mem}")
    headroom = max_mem // chunk_mem

    new_chunks = list(chunks)
    # only consolidate over these axes
    axes = sorted(chunk_limit_per_axis.keys())[::-1]
    for n_axis in axes:
        c_new = min(
            chunks[n_axis] * headroom, shape[n_axis], chunk_limit_per_axis[n_axis]
        )
        # print(f'  axis {n_axis}, {chunks[n_axis]} -> {c_new}')
        new_chunks[n_axis] = c_new
        chunk_mem = itemsize * prod(new_chunks)
        headroom = max_mem // chunk_mem

        if headroom == 1:
            break

    return tuple(new_chunks)


def _calculate_shared_chunks(
    read_chunks: Sequence[int], write_chunks: Sequence[int]
) -> Tuple[int, ...]:
    # Intermediate chunks are the smallest possible chunks which fit
    # into both read_chunks and write_chunks.
    # Example:
    #   read_chunks:            (20, 5)
    #   target_chunks:          (4, 25)
    #   intermediate_chunks:    (4, 5)
    # We don't need to check their memory usage: they are guaranteed to be smaller
    # than both read and write chunks.
    return tuple(
        min(c_read, c_target) for c_read, c_target in zip(read_chunks, write_chunks)
    )


def calculate_stage_chunks(
    read_chunks: Tuple[int, ...],
    write_chunks: Tuple[int, ...],
    stage_count: int = 1,
    epsilon: float = 1e-8,
) -> List[Tuple[int, ...]]:
    """
    Calculate chunks after each stage of a multi-stage rechunking.

    Each stage consists of "split" step followed by a "consolidate" step.

    The strategy used here is to progressively enlarge or shrink chunks along
    each dimension by the same multiple in each stage. This should roughly
    minimize the total number of arrays resulting from "split" steps in a
    multi-stage pipeline. It also keeps the total number of elements in each
    chunk constant, up to rounding error, so memory usage should remain
    constant.

    Examples::

        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=2)
        [(1000, 1000)]
        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=3)
        [(10000, 100), (100, 10000)]
        >>> calculate_stage_chunks((1_000_000, 1), (1, 1_000_000), stage_count=4)
        [(31623, 32), (1000, 1000), (32, 31623)]

    TODO: consider more sophisticated algorithms.
    """
    stage_chunks = []
    for stage in range(1, stage_count):
        power = stage / stage_count
        # Add a small floating-point epsilon so we don't inadvertently
        # round-down even chunk-sizes.
        chunks = tuple(
            floor(rc ** (1 - power) * wc ** power + epsilon)
            for rc, wc in zip(read_chunks, write_chunks)
        )
        stage_chunks.append(chunks)
    return stage_chunks


# not a tight upper bound, but ensures that the loop in
# multistage_rechunking_plan always terminates.
MAX_STAGES = 100


def multistage_rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    min_mem: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """A rechunking plan that can use multiple split/consolidate steps."""
    ndim = len(shape)
    if len(source_chunks) != ndim:
        raise ValueError(f"source_chunks {source_chunks} must have length {ndim}")
    if len(target_chunks) != ndim:
        raise ValueError(f"target_chunks {target_chunks} must have length {ndim}")

    source_chunk_mem = itemsize * prod(source_chunks)
    target_chunk_mem = itemsize * prod(target_chunks)

    if source_chunk_mem > max_mem:
        raise ValueError(
            f"Source chunk memory ({source_chunk_mem}) exceeds max_mem ({max_mem})"
        )
    if target_chunk_mem > max_mem:
        raise ValueError(
            f"Target chunk memory ({target_chunk_mem}) exceeds max_mem ({max_mem})"
        )

    if max_mem < min_mem:  # basic sanity test
        raise ValueError(
            "max_mem ({max_mem}) cannot be smaller than min_mem ({min_mem})"
        )

    if consolidate_writes:
        write_chunks = consolidate_chunks(shape, target_chunks, itemsize, max_mem)
    else:
        write_chunks = tuple(target_chunks)

    if consolidate_reads:
        read_chunk_limits: List[Optional[int]] = []
        for sc, wc in zip(source_chunks, write_chunks):
            limit: Optional[int]
            if wc > sc:
                # consolidate reads over this axis, up to the write chunk size
                limit = wc
            else:
                # don't consolidate reads over this axis
                limit = None
            read_chunk_limits.append(limit)

        read_chunks = consolidate_chunks(
            shape, source_chunks, itemsize, max_mem, read_chunk_limits
        )
    else:
        read_chunks = tuple(source_chunks)

    # increase the number of stages until min_mem is exceeded
    for stage_count in range(MAX_STAGES):

        stage_chunks = calculate_stage_chunks(read_chunks, write_chunks, stage_count)
        pre_chunks = [read_chunks] + stage_chunks
        post_chunks = stage_chunks + [write_chunks]

        int_chunks = [
            _calculate_shared_chunks(pre, post)
            for pre, post in zip(pre_chunks, post_chunks)
        ]
        if all(itemsize * prod(chunks) >= min_mem for chunks in int_chunks):
            # success!
            return list(zip(pre_chunks, int_chunks, post_chunks))

    raise AssertionError(
        "Failed to find a feasible multi-staging rechunking scheme satisfying "
        f"min_mem ({min_mem}) and max_mem ({max_mem}) constraints. "
        "Please file a bug report on GitHub: "
        "https://github.com/pangeo-data/rechunker/issues\n\n"
        "Include the following debugging info:\n"
        f"shape={shape}, source_chunks={source_chunks}, "
        f"target_chunks={target_chunks}, itemsize={itemsize}, "
        f"min_mem={min_mem}, max_mem={max_mem}, "
        f"consolidate_reads={consolidate_reads}, "
        f"consolidate_writes={consolidate_writes}"
    )


def rechunking_plan(
    shape: Sequence[int],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    itemsize: int,
    max_mem: int,
    consolidate_reads: bool = True,
    consolidate_writes: bool = True,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Calculate a plan for rechunking arrays.

    Parameters
    ----------
    shape : Tuple
        Array shape
    source_chunks : Tuple
        Original chunk shape (must be in form (5, 10, 20), no irregular chunks)
    target_chunks : Tuple
        Target chunk shape (must be in form (5, 10, 20), no irregular chunks)
    itemsize: int
        Number of bytes used to represent a single array element
    max_mem : Int
        Maximum permissible chunk memory size, measured in units of itemsize
    consolidate_reads: bool, optional
        Whether to apply read chunk consolidation
    consolidate_writes: bool, optional
        Whether to apply write chunk consolidation

    Returns
    -------
    new_chunks : tuple
        The new chunks, size guaranteed to be <= mam_mem
    """
    (stage,) = multistage_rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize=itemsize,
        min_mem=itemsize,
        max_mem=max_mem,
        consolidate_writes=consolidate_writes,
        consolidate_reads=consolidate_reads,
    )
    return stage
