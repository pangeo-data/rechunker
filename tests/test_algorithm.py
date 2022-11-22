#!/usr/bin/env python

"""Tests for `rechunker` package."""
import warnings
from math import prod
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

from rechunker.algorithm import (
    ExcessiveIOWarning,
    calculate_single_stage_io_ops,
    calculate_stage_chunks,
    consolidate_chunks,
    multistage_rechunking_plan,
    rechunking_plan,
)


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
        (32, (None, -1), (1, 8)),  # without limits
        (32, (None, 4), (1, 4)),  # with limits
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
    "shape",
    [
        (1000, 50, 1800, 3600),
    ],
)
@pytest.mark.parametrize(
    "chunks",
    [
        (1, 5, 1800, 3600),
    ],
)
@pytest.mark.parametrize(
    "itemsize",
    [
        4,
    ],
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


@pytest.mark.parametrize(
    "read_chunks, write_chunks, stage_count, expected",
    [
        ((100, 1), (1, 100), 1, []),
        ((100, 1), (1, 100), 2, [(10, 10)]),
        ((100, 1), (1, 100), 3, [(21, 4), (4, 21)]),
        ((1_000_000, 1), (1, 1_000_000), 2, [(1000, 1000)]),
        ((1_000_000, 1), (1, 1_000_000), 3, [(10000, 100), (100, 10000)]),
        ((1_000_000, 1), (1, 1_000_000), 4, [(31622, 31), (1000, 1000), (31, 31622)]),
        ((10, 10), (1, 100), 2, [(3, 31)]),
    ],
)
def test_calculate_stage_chunks(read_chunks, write_chunks, stage_count, expected):
    actual = calculate_stage_chunks(read_chunks, write_chunks, stage_count)
    assert actual == expected


@pytest.mark.parametrize(
    "shape, in_chunks, out_chunks, expected",
    [
        # simple 1d cases
        ((6,), (1,), (6,), 6),
        ((10,), (1,), (6,), 10),
        ((6,), (2,), (3,), 4),
        ((24,), (2,), (3,), 16),
        ((10,), (4,), (5,), 4),
        ((100,), (4,), (5,), 40),
        # simple 2d cases
        ((100, 100), (1, 100), (100, 1), 10_000),
        ((100, 100), (1, 10), (10, 1), 10_000),
        ((100, 100), (20, 20), (25, 25), 8**2),
        ((50, 50), (20, 20), (25, 25), 4**2),
        # edge cases where one chunk size is 43 (a prime)
        ((100,), (43,), (100,), 3),
        ((100,), (43,), (51,), 4),
        ((100,), (43,), (40,), 5),
        ((100,), (43,), (10,), 12),
        ((100,), (43,), (1,), 100),
    ],
)
def test_calculate_single_stage_io_ops(shape, in_chunks, out_chunks, expected):
    actual = calculate_single_stage_io_ops(shape, in_chunks, out_chunks)
    assert actual == expected


@st.composite
def io_ops_chunks(draw, max_len=1000):
    size = draw(st.integers(min_value=1, max_value=max_len))
    source = draw(st.integers(min_value=1, max_value=max_len))
    target = draw(st.integers(min_value=1, max_value=max_len))
    return (size, source, target)


@given(io_ops_chunks())
def test_calculate_single_stage_io_ops_hypothesis(inputs):
    size, source, target = inputs

    calculated = calculate_single_stage_io_ops((size,), (source,), (target,))

    table = np.empty(shape=(size, 2), dtype=int)
    for i in range(size):
        table[i, 0] = i // source
        table[i, 1] = i // target
    actual = np.unique(table, axis=0).shape[0]

    assert calculated == actual


def _verify_single_stage_plan_correctness(
    shape,
    source_chunks,
    read_chunks,
    int_chunks,
    write_chunks,
    target_chunks,
    itemsize,
    min_mem,
    max_mem,
):
    assert min_mem <= itemsize * prod(read_chunks) <= max_mem
    assert min_mem <= itemsize * prod(int_chunks) <= max_mem
    assert min_mem <= itemsize * prod(write_chunks) <= max_mem
    for n, sc, rc, ic, wc, tc in zip(
        shape, source_chunks, read_chunks, int_chunks, write_chunks, target_chunks
    ):
        # print(n, sc, rc, ic, wc, tc)
        assert rc >= sc  # read chunks bigger or equal to source chunks
        assert wc >= tc  # write chunks bigger or equal to target chunks
        # write chunks are either as big as the whole dimension or else
        # evenly slice the target chunks (avoid conflicts)
        assert (wc == n) or (wc % tc == 0)
        assert ic == min(rc, wc)  # intermediate chunks smaller than rear or write


def _verify_multistage_plan_correctness(
    shape,
    stages,
    source_chunks,
    target_chunks,
    itemsize,
    min_mem,
    max_mem,
    excessive_io=False,
):
    for sc, rc in zip(source_chunks, stages[0][0]):
        assert rc >= sc
    for n, tc, wc in zip(shape, target_chunks, stages[-1][-1]):
        assert wc >= tc
        assert (wc == n) or (wc % tc == 0)
    for read_chunks, int_chunks, write_chunks in stages:
        assert min_mem <= itemsize * prod(read_chunks) <= max_mem
        assert itemsize * prod(int_chunks) <= max_mem
        if excessive_io:
            assert min_mem >= itemsize * prod(int_chunks)
        else:
            assert min_mem <= itemsize * prod(int_chunks)
        assert min_mem <= itemsize * prod(write_chunks) <= max_mem
        for rc, ic, wc in zip(read_chunks, int_chunks, write_chunks):
            assert ic == min(rc, wc)


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
        ((16,), 4, (3,), (7,), 32, (7,), (7,), (7,)),  # uneven chunks
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
    min_mem = itemsize
    _verify_single_stage_plan_correctness(
        shape,
        source_chunks,
        read_chunks,
        int_chunks,
        write_chunks,
        target_chunks,
        itemsize,
        min_mem,
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
    min_mem = itemsize
    _verify_single_stage_plan_correctness(
        shape,
        source_chunks,
        read_chunks,
        int_chunks,
        write_chunks,
        target_chunks,
        itemsize,
        min_mem,
        max_mem,
    )


@pytest.mark.parametrize(
    "shape, source_chunks, target_chunks, itemsize, min_mem, max_mem, expected",
    [
        (
            (100, 100),
            (100, 1),
            (1, 100),
            1,
            1,
            100,
            [((100, 1), (1, 1), (1, 100))],
        ),
        (
            (100, 100),
            (100, 1),
            (1, 100),
            1,
            10,
            100,
            [
                ((100, 1), (10, 1), (10, 10)),
                ((10, 10), (1, 10), (1, 100)),
            ],
        ),
    ],
)
def test_multistage_rechunking_plan(
    shape,
    source_chunks,
    target_chunks,
    itemsize,
    min_mem,
    max_mem,
    expected,
):
    stages = multistage_rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, min_mem, max_mem
    )
    assert stages == expected


def test_multistage_rechunking_plan_warns():
    with pytest.warns(
        ExcessiveIOWarning,
        match="Search for multi-stage rechunking plan terminated",
    ):
        multistage_rechunking_plan((100, 100), (100, 1), (1, 100), 1, 90, 100)


@patch("rechunker.algorithm.MAX_STAGES", 1)
def test_multistage_rechunking_plan_fails():
    with pytest.raises(
        AssertionError,
        match="Failed to find a feasible multi-staging rechunking scheme",
    ):
        multistage_rechunking_plan((100, 100), (100, 1), (1, 100), 1, 10, 100)


def test_rechunking_plan_invalid_min_mem():
    with pytest.raises(
        ValueError,
        match="cannot be smaller than min_mem",
    ):
        multistage_rechunking_plan((100, 100), (100, 1), (1, 100), 1, 101, 100)


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
    orig_mem = max(source_chunk_mem, target_chunk_mem)
    return (tuple(shape), tuple(source_chunks), tuple(target_chunks), orig_mem)


@st.composite
def shapes_chunks_maxmem_for_ndim(draw):
    ndim = draw(st.integers(min_value=1, max_value=5))
    itemsize = 4
    shape, source_chunks, target_chunks, orig_mem = draw(
        shapes_chunks_maxmem(ndim=ndim, itemsize=4, max_len=10_000)
    )
    max_mem = orig_mem * 10
    min_mem = draw(
        st.integers(
            min_value=itemsize,
            max_value=min(itemsize * max(prod(shape) // 4, 1), 5 * orig_mem),
        )
    )
    return shape, source_chunks, target_chunks, min_mem, max_mem, itemsize


@given(shapes_chunks_maxmem_for_ndim())
def test_rechunking_plan_hypothesis(inputs):
    shape, source_chunks, target_chunks, min_mem, max_mem, itemsize = inputs
    print(shape, source_chunks, target_chunks, min_mem, max_mem)

    args = shape, source_chunks, target_chunks, itemsize, min_mem, max_mem
    with warnings.catch_warnings(record=True) as w_list:
        stages = multistage_rechunking_plan(*args)
        excessive_io = any(issubclass(w.category, ExcessiveIOWarning) for w in w_list)
    print(" plan: ", stages)

    # this should be guaranteed by the test
    source_chunk_mem = itemsize * prod(source_chunks)
    target_chunk_mem = itemsize * prod(target_chunks)
    assert source_chunk_mem <= max_mem
    assert target_chunk_mem <= max_mem

    ndim = len(shape)
    for stage in stages:
        read_chunks, int_chunks, write_chunks = stage
        assert len(read_chunks) == ndim
        assert len(int_chunks) == ndim
        assert len(write_chunks) == ndim

    _verify_multistage_plan_correctness(
        shape,
        stages,
        source_chunks,
        target_chunks,
        itemsize,
        min_mem,
        max_mem,
        excessive_io=excessive_io,
    )


# check for https://github.com/pangeo-data/rechunker/issues/115
def test_intermediate_to_target_memory():
    shape = (175320, 721, 1440)
    source_chunks = (24, 721, 1440)
    target_chunks = (21915, 103, 10)
    itemsize = 4
    max_mem = 12000000000  # 12 GB

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize,
        max_mem,
        consolidate_reads=True,
    )

    read_chunks2, int_chunks2, write_chunks2 = rechunking_plan(
        shape,
        int_chunks,
        target_chunks,
        itemsize,
        max_mem,
        consolidate_reads=True,
    )

    assert read_chunks2 == int_chunks2 == write_chunks2
