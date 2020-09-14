#!/usr/bin/env python

"""Tests for `rechunker` package."""
from rechunker.compat import prod

import pytest
from hypothesis import given, assume
import hypothesis.strategies as st

from rechunker.algorithm import consolidate_chunks, rechunking_plan


@pytest.mark.parametrize("shape, chunks", [((8, 8), (1, 2))])
@pytest.mark.parametrize(
    "itemsize, max_mem, expected",
    [
        (4, 8, (1, 2)),  # same chunks in and out
        (4, 16, (1, 4)),  # double chunks on axis 1
        (4, 17, (1, 4)),  # no difference
        (4, 64, (2, 8)),  # start on axis 0
        (4, 256, (8, 8)),  # maximum size
        (4, 512, (8, 8)),  # can't exceed total shape
        (8, 256, (4, 8)),  # can't exceed total shape
    ],
)
def test_consolidate_chunks(shape, chunks, itemsize, max_mem, expected):
    new_chunks = consolidate_chunks(shape, chunks, itemsize, max_mem)
    assert new_chunks == expected


@pytest.mark.parametrize("shape, chunks, itemsize", [((8, 8), (1, 2), 4)])
@pytest.mark.parametrize(
    "max_mem, chunk_limits, expected",
    [
        (16, (None, -1), (1, 4)),  # do last axis
        (16, (-1, None), (2, 2)),  # do first axis
        (32, (None, -1), (1, 8)),  # without limts
        (32, (None, 4), (1, 4)),  # with limts
        (32, (8, 4), (2, 4)),  # spill to next axis
        (32, (8, None), (4, 2)),
        (128, (10, None), (8, 2)),  # chunk_limit > shape truncated
    ],
)
def test_consolidate_chunks_w_limits(
    shape, chunks, itemsize, max_mem, chunk_limits, expected
):
    new_chunks = consolidate_chunks(
        shape, chunks, itemsize, max_mem, chunk_limits=chunk_limits
    )
    assert new_chunks == expected


def test_consolidate_chunks_mem_error():
    shape, chunks, itemsize = (8, 8), (1, 2), 4
    max_mem = 7
    with pytest.raises(ValueError, match=r"chunk_mem 8 > max_mem 7"):
        consolidate_chunks(shape, chunks, itemsize, max_mem)


@pytest.mark.parametrize("shape, chunks, itemsize, max_mem", [((8, 8), (1, 2), 4, 8)])
@pytest.mark.parametrize("chunk_limits", [(1, 1), (-2, 2)])
def test_consolidate_chunks_limit_error(shape, chunks, itemsize, max_mem, chunk_limits):
    with pytest.raises(ValueError, match=r"Invalid chunk_limits .*"):
        consolidate_chunks(shape, chunks, itemsize, max_mem, chunk_limits=chunk_limits)


@pytest.mark.parametrize(
    "shape", [(1000, 50, 1800, 3600),],
)
@pytest.mark.parametrize(
    "chunks", [(1, 5, 1800, 3600),],
)
@pytest.mark.parametrize(
    "itemsize", [4,],
)
@pytest.mark.parametrize(
    "max_mem, expected",
    [(1_000_000_000, (1, 35, 1800, 3600)), (3_000_000_000, (2, 50, 1800, 3600))],
)
def test_consolidate_chunks_4D(shape, chunks, itemsize, max_mem, expected):
    """A realistic example."""

    new_chunks = consolidate_chunks(shape, chunks, itemsize, max_mem)
    assert new_chunks == expected
    chunk_mem = itemsize * new_chunks[0] * new_chunks[1] * new_chunks[2] * new_chunks[3]
    assert chunk_mem <= max_mem


def _verify_plan_correctness(
    source_chunks,
    read_chunks,
    int_chunks,
    write_chunks,
    target_chunks,
    itemsize,
    max_mem,
):
    assert itemsize * prod(read_chunks) <= max_mem
    assert itemsize * prod(int_chunks) <= max_mem
    assert itemsize * prod(write_chunks) <= max_mem
    for sc, rc, ic, wc, tc in zip(
        source_chunks, read_chunks, int_chunks, write_chunks, target_chunks
    ):
        assert rc >= sc
        assert wc >= tc
        assert ic == min(rc, wc)
        # todo: check for write overlaps


@pytest.mark.parametrize(
    (
        "shape, itemsize, source_chunks, target_chunks, max_mem, read_chunks_expected, "
        "intermediate_chunks_expected, write_chunks_expected"
    ),
    [
        ((8,), 4, (1,), (1,), 4, (1,), (1,), (1,)),  # pass chunks through unchanged
        ((8,), 4, (1,), (1,), 8, (2,), (2,), (2,)),  # consolidate reading and writing
        ((8,), 4, (1,), (2,), 8, (2,), (2,), (2,)),
        ((8,), 4, (1,), (2,), 16, (4,), (4,), (4,)),  # consolidate
        ((8,), 4, (1,), (2,), 17, (4,), (4,), (4,)),  # no difference
        ((16,), 4, (3,), (7,), 32, (6,), (6,), (7,)),  # uneven chunks
    ],
)
def test_rechunking_plan_1D(
    shape,
    source_chunks,
    target_chunks,
    itemsize,
    max_mem,
    read_chunks_expected,
    intermediate_chunks_expected,
    write_chunks_expected,
):
    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )
    assert read_chunks == read_chunks_expected
    assert int_chunks == intermediate_chunks_expected
    assert write_chunks == write_chunks_expected
    _verify_plan_correctness(
        source_chunks,
        read_chunks,
        int_chunks,
        write_chunks,
        target_chunks,
        itemsize,
        max_mem,
    )


@pytest.mark.parametrize(
    "shape, source_chunks, target_chunks, itemsize", [((8, 8), (1, 8), (8, 1), 4)]
)
@pytest.mark.parametrize(
    "max_mem, read_chunks_expected, intermediate_chunks_expected, write_chunks_expected",
    [
        (32, (1, 8), (1, 1), (8, 1)),  # no consolidation possible
        (64, (2, 8), (2, 2), (8, 2)),  # consolidate 1->2 on read / write
        (256, (8, 8), (8, 8), (8, 8)),  # full consolidation
        (512, (8, 8), (8, 8), (8, 8)),  # more memory doesn't help
    ],
)
def test_rechunking_plan_2d(
    shape,
    source_chunks,
    target_chunks,
    itemsize,
    max_mem,
    read_chunks_expected,
    intermediate_chunks_expected,
    write_chunks_expected,
):
    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )
    assert read_chunks == read_chunks_expected
    assert int_chunks == intermediate_chunks_expected
    assert write_chunks == write_chunks_expected
    _verify_plan_correctness(
        source_chunks,
        read_chunks,
        int_chunks,
        write_chunks,
        target_chunks,
        itemsize,
        max_mem,
    )


@st.composite
def shapes_chunks_maxmem(draw, ndim=3, itemsize=4, max_len=10_000):
    """Generate the data we need to test rechunking_plan."""
    shape = []
    source_chunks = []
    target_chunks = []
    for n in range(ndim):
        sh = draw(st.integers(min_value=1, max_value=max_len))
        sc = draw(st.integers(min_value=1, max_value=max_len))
        tc = draw(st.integers(min_value=1, max_value=max_len))
        assume(sc <= sh)
        assume(tc <= sh)
        shape.append(sh)
        source_chunks.append(sc)
        target_chunks.append(tc)
    source_chunk_mem = itemsize * prod(source_chunks)
    target_chunk_mem = itemsize * prod(target_chunks)
    min_mem = max(source_chunk_mem, target_chunk_mem)
    return (tuple(shape), tuple(source_chunks), tuple(target_chunks), min_mem)


@st.composite
def shapes_chunks_maxmem_for_ndim(draw):
    ndim = draw(st.integers(min_value=1, max_value=5))
    itemsize = 4
    shape, source_chunks, target_chunks, min_mem = draw(
        shapes_chunks_maxmem(ndim=ndim, itemsize=4, max_len=10_000)
    )
    max_mem = min_mem * 10
    return shape, source_chunks, target_chunks, max_mem, itemsize


@given(shapes_chunks_maxmem_for_ndim())
def test_rechunking_plan_hypothesis(inputs):
    shape, source_chunks, target_chunks, max_mem, itemsize = inputs
    # print(shape, source_chunks, target_chunks, max_mem)

    args = shape, source_chunks, target_chunks, itemsize, max_mem
    read_chunks, int_chunks, write_chunks = rechunking_plan(*args)
    # print(" plan: ", read_chunks, int_chunks, write_chunks)

    # this should be guaranteed by the test
    source_chunk_mem = itemsize * prod(source_chunks)
    target_chunk_mem = itemsize * prod(target_chunks)
    assert source_chunk_mem <= max_mem
    assert target_chunk_mem <= max_mem

    ndim = len(shape)
    assert len(read_chunks) == ndim
    assert len(int_chunks) == ndim
    assert len(write_chunks) == ndim

    _verify_plan_correctness(
        source_chunks,
        read_chunks,
        int_chunks,
        write_chunks,
        target_chunks,
        itemsize,
        max_mem,
    )
