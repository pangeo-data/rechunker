#!/usr/bin/env python

"""Tests for `zarr_rechunker` package."""

import pytest
from zarr_rechunker.zarr_rechunker import consolidate_chunks


@pytest.mark.parametrize("shape, chunks", [((8, 8), (1, 2))])
@pytest.mark.parametrize("itemsize, max_mem, expected",
                         [(4, 8, (1, 2)), # same chunks in and out
                          (4, 16, (1, 4)), # double chunks on axis 1
                          (4, 17, (1, 4)), # no difference
                          (4, 64, (2, 8)), # start on axis 0
                          (4, 256, (8, 8)), # maximum size
                          (4, 512, (8, 8)), # can't exceed total shape
                          (8, 256, (4, 8)), # can't exceed total shape
                         ])
def test_consolidate_chunks(shape, chunks, itemsize, max_mem, expected):
    new_chunks = consolidate_chunks(shape, chunks, itemsize, max_mem)
    assert new_chunks == expected


@pytest.mark.parametrize("shape, chunks, itemsize", [((8, 8), (1, 2), 4)])
@pytest.mark.parametrize("max_mem, chunk_limits, expected",
                         [(16, (None, -1), (1, 4)), # do last axis
                          (16, (-1, None), (2, 2)), # do first axis
                          (32, (None, -1), (1, 8)), # without limts
                          (32, (None, 4), (1, 4)), # with limts
                          (32, (8, 4), (2, 4)), # spill to next axis
                          (32, (8, None), (4, 2)),
                          (128, (10, None), (8, 2)), # chunk_limit > shape truncated
                         ])
def test_consolidate_chunks_w_limits(shape, chunks, itemsize, max_mem, chunk_limits, expected):
    new_chunks = consolidate_chunks(shape, chunks, itemsize, max_mem,
                                    chunk_limits=chunk_limits)
    assert new_chunks == expected


def test_consolidate_chunks_mem_error():
    shape, chunks, itemsize = (8, 8), (1, 2), 4
    max_mem = 7
    with pytest.raises(ValueError, match=r"chunk_mem 8 > max_mem 7"):
        consolidate_chunks(shape, chunks, itemsize, max_mem)


@pytest.mark.parametrize("shape, chunks, itemsize, max_mem",
                         [((8, 8), (1, 2), 4,  8)])
@pytest.mark.parametrize("chunk_limits", [(1, 1), (-2, 2)])
def test_consolidate_chunks_limit_error(shape, chunks, itemsize, max_mem, chunk_limits):
    with pytest.raises(ValueError, match=r'Invalid chunk_limits .*'):
        consolidate_chunks(shape, chunks, itemsize, max_mem,
                           chunk_limits=chunk_limits)


@pytest.mark.parametrize("shape", [(1000, 50, 1800, 3600),])
@pytest.mark.parametrize("chunks", [(1, 5, 1800, 3600),])
@pytest.mark.parametrize("itemsize", [4,])
@pytest.mark.parametrize("max_mem, expected",
                         [(1_000_000_000, (1, 35, 1800, 3600)),
                          (3_000_000_000, (2, 50, 1800, 3600))])
def test_consolidate_chunks_4D(shape, chunks, itemsize, max_mem, expected):
    """A realistic example."""

    new_chunks = consolidate_chunks(shape, chunks, itemsize, max_mem)
    assert new_chunks == expected
    chunk_mem = itemsize * new_chunks[0]* new_chunks[1] * new_chunks[2] * new_chunks[3]
    assert itemsize <= max_mem
