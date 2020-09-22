from functools import partial
import importlib
import pytest

from pathlib import Path
import zarr
import dask.array as dsa
import dask
import dask.core
import xarray
import numpy

from rechunker import api


_DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def requires_import(module, *args):
    try:
        importlib.import_module(module)
    except ImportError:
        skip = True
    else:
        skip = False
    mark = pytest.mark.skipif(skip, reason=f"requires {module}")
    return pytest.param(*args, marks=mark)


requires_beam = partial(requires_import, "apache_beam")
requires_prefect = partial(requires_import, "prefect")
requires_pywren = partial(requires_import, "pywren_ibm_cloud")


@pytest.fixture(params=[(8000, 200), {"y": 8000, "x": 200}])
def target_chunks(request):
    return request.param


def test_invalid_executor():
    with pytest.raises(ValueError, match="unrecognized executor"):
        api._get_executor("unknown")


@pytest.mark.parametrize("shape", [(100, 50)])
@pytest.mark.parametrize("source_chunks", [(10, 50)])
@pytest.mark.parametrize("target_chunks", [(20, 10)])
@pytest.mark.parametrize("max_mem", ["10MB"])
@pytest.mark.parametrize("pass_temp", [True, False])
@pytest.mark.parametrize("executor", ["dask", api._get_executor("dask")])
def test_rechunk_dataset(
    tmp_path, shape, source_chunks, target_chunks, max_mem, pass_temp, executor
):
    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")

    a = numpy.arange(numpy.prod(shape)).reshape(shape).astype("f4")
    a[-1] = numpy.nan
    ds = xarray.Dataset(
        dict(
            a=xarray.DataArray(
                a, dims=["x", "y"], attrs={"a1": 1, "a2": [1, 2, 3], "a3": "x"}
            ),
            b=xarray.DataArray(numpy.ones(shape[0]), dims=["x"]),
            c=xarray.DataArray(numpy.ones(shape[1]), dims=["y"]),
        ),
        attrs={"a1": 1, "a2": [1, 2, 3], "a3": "x"},
    )
    ds = ds.chunk(chunks=dict(zip(["x", "y"], source_chunks)))
    encoding = dict(
        a=dict(
            chunks=target_chunks,
            compressor=zarr.Blosc(cname="zstd"),
            dtype="int32",
            scale_factor=0.1,
            _FillValue=-9999,
        ),
        b=dict(chunks=target_chunks[:1]),
    )
    rechunked = api.rechunk_dataset(
        ds,
        encoding=encoding,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store if pass_temp else None,
        executor=executor,
    )
    assert isinstance(rechunked, api.Rechunked)
    rechunked.execute()

    # Validate encoded variables
    dst = xarray.open_zarr(target_store, decode_cf=False)
    assert dst.a.dtype == encoding["a"]["dtype"]
    assert all(dst.a.values[-1] == encoding["a"]["_FillValue"])

    # Validate decoded variables
    dst = xarray.open_zarr(target_store, decode_cf=True)
    assert dst.a.data.chunksize == target_chunks
    assert dst.b.data.chunksize == target_chunks[:1]
    assert dst.c.data.chunksize == source_chunks[1:]
    xarray.testing.assert_equal(ds.compute(), dst.compute())


@pytest.mark.parametrize("shape", [(8000, 8000)])
@pytest.mark.parametrize("source_chunks", [(200, 8000)])
@pytest.mark.parametrize("dtype", ["f4"])
@pytest.mark.parametrize("max_mem", [25600000, "25.6MB"])
@pytest.mark.parametrize(
    "executor",
    [
        "dask",
        "python",
        requires_beam("beam"),
        requires_prefect("prefect"),
        requires_pywren("pywren"),
    ],
)
@pytest.mark.parametrize(
    "dims,target_chunks",
    [
        (None, (8000, 200)),
        # would be nice to support this syntax eventually
        pytest.param(None, (-1, 200), marks=pytest.mark.xfail),
        (["y", "x"], (8000, 200)),
        (["y", "x"], {"y": 8000, "x": 200}),
        # can't infer missing dimension chunk specification
        pytest.param(["y", "x"], {"x": 200}, marks=pytest.mark.xfail),
        # can't use dict syntax without array dims
        pytest.param(None, {"y": 8000, "x": 200}, marks=pytest.mark.xfail),
    ],
)
def test_rechunk_array(
    tmp_path, shape, source_chunks, dtype, dims, target_chunks, max_mem, executor
):

    ### Create source array ###
    store_source = str(tmp_path / "source.zarr")
    source_array = zarr.ones(
        shape, chunks=source_chunks, dtype=dtype, store=store_source
    )
    # add some attributes
    source_array.attrs["foo"] = "bar"
    if dims:
        source_array.attrs[_DIMENSION_KEY] = dims

    ### Create targets ###
    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")

    rechunked = api.rechunk(
        source_array,
        target_chunks,
        max_mem,
        target_store,
        temp_store=temp_store,
        executor=executor,
    )
    assert isinstance(rechunked, api.Rechunked)

    target_array = zarr.open(target_store)

    if isinstance(target_chunks, dict):
        target_chunks_list = [target_chunks[d] for d in dims]
    else:
        target_chunks_list = target_chunks
    assert target_array.chunks == tuple(target_chunks_list)
    assert dict(source_array.attrs) == dict(target_array.attrs)

    result = rechunked.execute()
    assert isinstance(result, zarr.Array)
    a_tar = dsa.from_zarr(target_array)
    assert dsa.equal(a_tar, 1).all().compute()


@pytest.mark.parametrize("shape", [(8000, 8000)])
@pytest.mark.parametrize("source_chunks", [(200, 8000), (800, 8000)])
@pytest.mark.parametrize("dtype", ["f4"])
@pytest.mark.parametrize("max_mem", [25600000])
@pytest.mark.parametrize(
    "target_chunks", [(200, 8000), (800, 8000), (8000, 200), (400, 8000),],
)
def test_rechunk_dask_array(
    tmp_path, shape, source_chunks, dtype, target_chunks, max_mem
):

    ### Create source array ###
    source_array = dsa.ones(shape, chunks=source_chunks, dtype=dtype)

    ### Create targets ###
    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")

    rechunked = api.rechunk(
        source_array, target_chunks, max_mem, target_store, temp_store=temp_store
    )
    assert isinstance(rechunked, api.Rechunked)

    target_array = zarr.open(target_store)

    assert target_array.chunks == tuple(target_chunks)

    result = rechunked.execute()
    assert isinstance(result, zarr.Array)
    a_tar = dsa.from_zarr(target_array)
    assert dsa.equal(a_tar, 1).all().compute()


@pytest.mark.parametrize(
    "executor",
    [
        "dask",
        "python",
        requires_beam("beam"),
        requires_prefect("prefect"),
        requires_pywren("pywren"),
    ],
)
def test_rechunk_group(tmp_path, executor):
    store_source = str(tmp_path / "source.zarr")
    group = zarr.group(store_source)
    group.attrs["foo"] = "bar"
    # 800 byte chunks
    a = group.ones("a", shape=(5, 10, 20), chunks=(1, 10, 20), dtype="f4")
    a.attrs["foo"] = "bar"
    b = group.ones("b", shape=(20,), chunks=(10,), dtype="f4")
    b.attrs["foo"] = "bar"

    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")

    max_mem = 1600  # should force a two-step plan for a
    target_chunks = {"a": (5, 10, 4), "b": (20,)}

    rechunked = api.rechunk(
        group,
        target_chunks,
        max_mem,
        target_store,
        temp_store=temp_store,
        executor=executor,
    )
    assert isinstance(rechunked, api.Rechunked)

    target_group = zarr.open(target_store)
    assert "a" in target_group
    assert "b" in target_group
    assert dict(group.attrs) == dict(target_group.attrs)

    rechunked.execute()
    for aname in target_chunks:
        a_tar = dsa.from_zarr(target_group[aname])
        assert dsa.equal(a_tar, 1).all().compute()


@pytest.fixture(params=["Array", "Group"])
def rechunked_fn(tmp_path, request):
    if request.param == "Group":
        store_source = str(tmp_path / "source.zarr")
        group = zarr.group(store_source)
        group.attrs["foo"] = "bar"
        # 800 byte chunks
        a = group.ones("a", shape=(5, 10, 20), chunks=(1, 10, 20), dtype="f4")
        a.attrs["foo"] = "bar"
        b = group.ones("b", shape=(8000,), chunks=(100,), dtype="f4")
        b.attrs["foo"] = "bar"

        target_store = str(tmp_path / "target.zarr")
        temp_store = str(tmp_path / "temp.zarr")

        max_mem = 16000  # should force a two-step plan for b
        target_chunks = {"a": (5, 10, 4), "b": (4000,)}

        rechunked_fn = partial(
            api.rechunk,
            group,
            target_chunks,
            max_mem,
            target_store,
            temp_store=temp_store,
        )
    else:
        shape = (8000, 8000)
        source_chunks = (200, 8000)
        dtype = "f4"
        max_mem = 25600000
        dims = None
        target_chunks = (8000, 200)

        store_source = str(tmp_path / "source.zarr")
        source_array = zarr.ones(
            shape, chunks=source_chunks, dtype=dtype, store=store_source
        )
        # add some attributes
        source_array.attrs["foo"] = "bar"
        if dims:
            source_array.attrs[_DIMENSION_KEY] = dims

        ### Create targets ###
        target_store = str(tmp_path / "target.zarr")
        temp_store = str(tmp_path / "temp.zarr")

        rechunked_fn = partial(
            api.rechunk,
            source_array,
            target_chunks,
            max_mem,
            target_store,
            temp_store=temp_store,
        )
    return rechunked_fn


@pytest.fixture()
def rechunked(rechunked_fn):
    return rechunked_fn()


def test_repr(rechunked):
    assert isinstance(rechunked, api.Rechunked)
    repr_str = repr(rechunked)

    assert repr_str.startswith("<Rechunked>")
    assert all(thing in repr_str for thing in ["Source", "Intermediate", "Target"])


def test_rechunk_option_overwrite(rechunked_fn):
    rechunked_fn().execute()
    # TODO: make this match more reliable based on outcome of
    # https://github.com/zarr-developers/zarr-python/issues/605
    with pytest.raises(ValueError, match=r"path .* contains an array"):
        rechunked_fn().execute()
    rechunked = rechunked_fn(
        temp_options=dict(overwrite=True), target_options=dict(overwrite=True)
    )
    rechunked.execute()


def test_rechunk_option_compression(rechunked_fn):
    def rechunk(compressor):
        rechunked = rechunked_fn(
            temp_options=dict(overwrite=True, compressor=compressor),
            target_options=dict(overwrite=True, compressor=compressor),
        )
        rechunked.execute()
        return sum(
            file.stat().st_size
            for file in Path(rechunked._target.store.path).rglob("*")
        )

    size_uncompressed = rechunk(None)
    size_compressed = rechunk(
        zarr.Blosc(cname="zstd", clevel=9, shuffle=zarr.Blosc.SHUFFLE)
    )
    assert size_compressed < size_uncompressed


def test_rechunk_reserved_option(rechunked_fn):
    for o in ["shape", "chunks", "dtype", "store", "name"]:
        with pytest.raises(
            ValueError, match=f"Optional array arguments must not include {o}"
        ):
            rechunked_fn(temp_options={o: True})
        with pytest.raises(
            ValueError, match=f"Optional array arguments must not include {o}"
        ):
            rechunked_fn(target_options={o: True})


def test_repr_html(rechunked):
    rechunked._repr_html_()  # no exceptions


def test_no_intermediate():
    a = zarr.ones((4, 4), chunks=(2, 2))
    b = zarr.ones((4, 4), chunks=(4, 1))
    rechunked = api.Rechunked(None, None, source=a, intermediate=None, target=b)
    assert "Intermediate" not in repr(rechunked)
    rechunked._repr_html_()


def test_no_intermediate_fused(tmp_path):
    shape = (8000, 8000)
    source_chunks = (200, 8000)
    dtype = "f4"
    max_mem = 25600000
    target_chunks = (400, 8000)

    store_source = str(tmp_path / "source.zarr")
    source_array = zarr.ones(
        shape, chunks=source_chunks, dtype=dtype, store=store_source
    )

    target_store = str(tmp_path / "target.zarr")

    rechunked = api.rechunk(source_array, target_chunks, max_mem, target_store)

    num_tasks = len([v for v in rechunked.plan.dask.values() if dask.core.istask(v)])
    assert num_tasks < 20  # less than if no fuse


def test_pywren_function_executor(tmp_path):
    pytest.importorskip("pywren_ibm_cloud")
    from rechunker.executors.pywren import (
        pywren_local_function_executor,
        PywrenExecutor,
    )

    # Create a Pywren function exectutor that we manage ourselves
    # and pass in to rechunker's PywrenExecutor
    with pywren_local_function_executor() as function_executor:

        executor = PywrenExecutor(function_executor)

        shape = (8000, 8000)
        source_chunks = (200, 8000)
        dtype = "f4"
        max_mem = 25600000
        target_chunks = (400, 8000)

        ### Create source array ###
        store_source = str(tmp_path / "source.zarr")
        source_array = zarr.ones(
            shape, chunks=source_chunks, dtype=dtype, store=store_source
        )

        ### Create targets ###
        target_store = str(tmp_path / "target.zarr")
        temp_store = str(tmp_path / "temp.zarr")

        rechunked = api.rechunk(
            source_array,
            target_chunks,
            max_mem,
            target_store,
            temp_store=temp_store,
            executor=executor,
        )
        assert isinstance(rechunked, api.Rechunked)

        target_array = zarr.open(target_store)

        assert target_array.chunks == tuple(target_chunks)

        result = rechunked.execute()
        assert isinstance(result, zarr.Array)
        a_tar = dsa.from_zarr(target_array)
        assert dsa.equal(a_tar, 1).all().compute()
