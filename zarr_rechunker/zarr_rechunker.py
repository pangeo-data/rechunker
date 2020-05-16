"""Main module."""


import math
from math import prod
from functools import reduce

import numpy as np # type: ignore

from typing import Callable, Iterable, Sequence, Union, Optional, List, Tuple


def consolidate_chunks(shape: Sequence[int],
                       chunks: Sequence[int],
                       itemsize: int,
                       max_mem: int,
                       axes: Optional[Sequence[int]]=None,
                       chunk_limits: Optional[Sequence[int]]=None) -> Sequence[int]:
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
    axes : Iterable, optional
        List of axes on which to consolidate. Otherwise all.
    chunk_limits : Tuple, optional
        Maximum size of each chunk along each axis.

    Returns
    -------
    new_chunks : tuple
        The new chunks, size guaranteed to be <= mam_mem
    """

    if axes is None:
        axes = list(range(len(shape)))
    if chunk_limits is None:
        chunk_limits = shape

    if any([cl < cs for cl, cs in zip(chunk_limits, chunks)]):
        raise ValueError(f"chunk_limits {chunk_limits} are smaller than chunks {chunks}")

    chunk_mem = itemsize * prod(chunks)
    if chunk_mem > max_mem:
        raise ValueError(f"chunk_mem {chunk_mem} > max_mem {max_mem}")
    headroom = max_mem // chunk_mem

    new_chunks = list(chunks)

    for n_axis in sorted(axes)[::-1]:

        c_new= min(chunks[n_axis] * headroom, shape[n_axis], chunk_limits[n_axis])
        print(f'  axis {n_axis}, {chunks[n_axis]} -> {c_new}')
        new_chunks[n_axis] = c_new
        chunk_mem = itemsize * prod(new_chunks)
        headroom = max_mem // chunk_mem

        if headroom == 1:
            break

    return tuple(new_chunks)
