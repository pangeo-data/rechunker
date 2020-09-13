"""Core rechunking algorithm stuff."""
from typing import Sequence, Optional, List, Tuple

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

    if consolidate_writes:
        write_chunks = consolidate_chunks(shape, target_chunks, itemsize, max_mem)
    else:
        write_chunks = tuple(target_chunks)

    if consolidate_reads:
        read_chunk_limits: List[Optional[int]]
        read_chunk_limits = []  #
        for n_ax, (sc, wc) in enumerate(zip(source_chunks, write_chunks)):
            read_chunk_lim: Optional[int]
            if wc > sc:
                # consolidate reads over this axis, up to the write chunk size
                read_chunk_lim = wc
            else:
                # don't consolidate reads over this axis
                read_chunk_lim = None
            read_chunk_limits.append(read_chunk_lim)

        read_chunks = consolidate_chunks(
            shape, source_chunks, itemsize, max_mem, read_chunk_limits
        )
    else:
        read_chunks = tuple(source_chunks)

    # Intermediate chunks  are the smallest possible chunks which fit
    # into both read_chunks and write_chunks.
    # Example:
    #   read_chunks:            (20, 5)
    #   target_chunks:          (4, 25)
    #   intermediate_chunks:    (4, 5)
    # We don't need to check their memory usage: they are guaranteed to be smaller
    # than both read and write chunks.
    intermediate_chunks = [
        min(c_read, c_target) for c_read, c_target in zip(read_chunks, write_chunks)
    ]

    return read_chunks, tuple(intermediate_chunks), write_chunks
