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
import fsspec

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
@pytest.mark.parametrize(
    "target_chunks",
    [{"a": (20, 10), "b": (20,)}, {"a": {"x": 20, "y": 10}, "b": {"x": 20}}],
)
@pytest.mark.parametrize("max_mem", ["10MB"])
@pytest.mark.parametrize("executor", ["dask"])
@pytest.mark.parametrize("target_store", ["target.zarr", "mapper.target.zarr"])
@pytest.mark.parametrize("temp_store", ["temp.zarr", "mapper.temp.zarr"])
def test_rechunk_dataset(
    tmp_path,
    shape,
    source_chunks,
    target_chunks,
    max_mem,
    executor,
    target_store,
    temp_store,
):
    if target_store.startswith("mapper"):
        target_store = fsspec.get_mapper(str(tmp_path) + target_store)
        temp_store = fsspec.get_mapper(str(tmp_path) + temp_store)
    else:
        target_store = str(tmp_path / target_store)
        temp_store = str(tmp_path / temp_store)

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
        coords=dict(
            cx=xarray.DataArray(numpy.ones(shape[0]), dims=["x"]),
            cy=xarray.DataArray(numpy.ones(shape[1]), dims=["y"]),
        ),
        attrs={"a1": 1, "a2": [1, 2, 3], "a3": "x"},
    )
    ds = ds.chunk(chunks=dict(zip(["x", "y"], source_chunks)))
    options = dict(
        a=dict(
            compressor=zarr.Blosc(cname="zstd"),
            dtype="int32",
            scale_factor=0.1,
            _FillValue=-9999,
        )
    )
    rechunked = api.rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        target_options=options,
        temp_store=temp_store,
        executor=executor,
    )
    assert isinstance(rechunked, api.Rechunked)
    rechunked.execute()

    # Validate encoded variables
    dst = xarray.open_zarr(target_store, decode_cf=False)
    assert dst.a.dtype == options["a"]["dtype"]
    assert all(dst.a.values[-1] == options["a"]["_FillValue"])
    assert dst.a.encoding["compressor"] is not None

    # Validate decoded variables
    dst = xarray.open_zarr(target_store, decode_cf=True)
    target_chunks_expected = (
        target_chunks["a"]
        if isinstance(target_chunks["a"], tuple)
        else (target_chunks["a"]["x"], target_chunks["a"]["y"])
    )
    assert dst.a.data.chunksize == target_chunks_expected
    assert dst.b.data.chunksize == target_chunks_expected[:1]
    assert dst.c.data.chunksize == source_chunks[1:]
    xarray.testing.assert_equal(ds.compute(), dst.compute())
    assert ds.attrs == dst.attrs


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
@pytest.mark.parametrize("source_store", ["source.zarr", "mapper.source.zarr"])
@pytest.mark.parametrize("target_store", ["target.zarr", "mapper.target.zarr"])
@pytest.mark.parametrize("temp_store", ["temp.zarr", "mapper.temp.zarr"])
def test_rechunk_group(tmp_path, executor, source_store, target_store, temp_store):
    if source_store.startswith("mapper"):
        store_source = fsspec.get_mapper(str(tmp_path) + source_store)
        target_store = fsspec.get_mapper(str(tmp_path) + target_store)
        temp_store = fsspec.get_mapper(str(tmp_path) + temp_store)
    else:
        store_source = str(tmp_path / source_store)
        target_store = str(tmp_path / target_store)
        temp_store = str(tmp_path / temp_store)

    group = zarr.group(store_source)
    group.attrs["foo"] = "bar"
    # 800 byte chunks
    a = group.ones("a", shape=(5, 10, 20), chunks=(1, 10, 20), dtype="f4")
    a.attrs["foo"] = "bar"
    b = group.ones("b", shape=(20,), chunks=(10,), dtype="f4")
    b.attrs["foo"] = "bar"

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
        assert target_group[aname].chunks == target_chunks[aname]
        a_tar = dsa.from_zarr(target_group[aname])
        assert dsa.equal(a_tar, 1).all().compute()


def sample_xarray_dataset():
    return xarray.Dataset(
        dict(
            a=xarray.DataArray(
                dsa.ones(shape=(10, 20, 40), chunks=(5, 10, 4), dtype="f4"),
                dims=("x", "y", "z"),
                attrs={"foo": "bar"},
            ),
            b=xarray.DataArray(
                dsa.ones(shape=(8000,), chunks=(200,), dtype="f4"),
                dims="w",
                attrs={"foo": "bar"},
            ),
        ),
        attrs={"foo": "bar"},
    )


def sample_zarr_group(tmp_path):
    path = str(tmp_path / "source.zarr")
    group = zarr.group(path)
    group.attrs["foo"] = "bar"
    # 800 byte chunks
    a = group.ones("a", shape=(10, 20, 40), chunks=(5, 10, 4), dtype="f4")
    a.attrs["foo"] = "bar"
    b = group.ones("b", shape=(8000,), chunks=(200,), dtype="f4")
    b.attrs["foo"] = "bar"
    return group


def sample_zarr_array(tmp_path):
    shape = (8000, 8000)
    source_chunks = (200, 8000)
    dtype = "f4"
    dims = None

    path = str(tmp_path / "source.zarr")
    array = zarr.ones(shape, chunks=source_chunks, dtype=dtype, store=path)
    # add some attributes
    array.attrs["foo"] = "bar"
    if dims:
        array.attrs[_DIMENSION_KEY] = dims
    return array


@pytest.fixture(params=["Array", "Group", "Dataset"])
def rechunk_args(tmp_path, request):
    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")
    max_mem = 1600  # should force a two-step plan for a and b
    target_chunks = {"a": (10, 5, 4), "b": (100,)}

    args = dict(
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )
    if request.param == "Dataset":
        ds = sample_xarray_dataset()
        args.update({"source": ds})
    elif request.param == "Group":
        group = sample_zarr_group(tmp_path)
        args.update({"source": group})
    else:
        array = sample_zarr_array(tmp_path)
        max_mem = 25600000
        target_chunks = (8000, 200)

        args.update(
            {"source": array, "target_chunks": target_chunks, "max_mem": max_mem,}
        )
    return args


@pytest.fixture()
def rechunked(rechunk_args):
    return api.rechunk(**rechunk_args)


def test_repr(rechunked):
    assert isinstance(rechunked, api.Rechunked)
    repr_str = repr(rechunked)

    assert repr_str.startswith("<Rechunked>")
    assert all(thing in repr_str for thing in ["Source", "Intermediate", "Target"])


def test_repr_html(rechunked):
    rechunked._repr_html_()  # no exceptions


def _is_collection(source):
    assert isinstance(
        source,
        (dask.array.Array, zarr.core.Array, zarr.hierarchy.Group, xarray.Dataset),
    )
    return isinstance(source, (zarr.hierarchy.Group, xarray.Dataset))


def _wrap_options(source, options):
    if _is_collection(source):
        options = {v: options for v in source}
    return options


def test_rechunk_option_overwrite(rechunk_args):
    api.rechunk(**rechunk_args).execute()
    # TODO: make this match more reliable based on outcome of
    # https://github.com/zarr-developers/zarr-python/issues/605
    with pytest.raises(ValueError, match=r"path .* contains an array"):
        api.rechunk(**rechunk_args).execute()
    options = _wrap_options(rechunk_args["source"], dict(overwrite=True))
    api.rechunk(**rechunk_args, target_options=options).execute()


def test_rechunk_passthrough(rechunk_args):
    # Verify that no errors are raised when the target chunks == source chunks
    if _is_collection(rechunk_args["source"]):
        rechunk_args["target_chunks"] = {v: None for v in rechunk_args["source"]}
    else:
        rechunk_args["target_chunks"] = None
    api.rechunk(**rechunk_args).execute()


def test_rechunk_no_temp_dir_provided_error(rechunk_args):
    # Verify that the correct error is raised when no temp_store is given
    # and the chunks to write differ from the chunks to read
    args = {k: v for k, v in rechunk_args.items() if k != "temp_store"}
    with pytest.raises(ValueError, match="A temporary store location must be provided"):
        api.rechunk(**args).execute()


def test_rechunk_option_compression(rechunk_args):
    def rechunk(compressor):
        options = _wrap_options(
            rechunk_args["source"], dict(overwrite=True, compressor=compressor)
        )
        rechunked = api.rechunk(**rechunk_args, target_options=options)
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


def test_rechunk_invalid_option(rechunk_args):
    if isinstance(rechunk_args["source"], xarray.Dataset):
        # Options are essentially unbounded for Xarray (for CF encoding params),
        # so check only options with special error cases
        options = _wrap_options(rechunk_args["source"], {"chunks": 10})
        with pytest.raises(
            ValueError,
            match="Chunks must be provided in ``target_chunks`` rather than options",
        ):
            api.rechunk(**rechunk_args, target_options=options)
    else:
        for o in ["shape", "chunks", "dtype", "store", "name", "unknown"]:
            options = _wrap_options(rechunk_args["source"], {o: True})
            with pytest.raises(ValueError, match=f"Zarr options must not include {o}"):
                api.rechunk(**rechunk_args, temp_options=options)
            with pytest.raises(ValueError, match=f"Zarr options must not include {o}"):
                api.rechunk(**rechunk_args, target_options=options)


def test_rechunk_bad_target_chunks(rechunk_args):
    if not _is_collection(rechunk_args["source"]):
        return
    rechunk_args = dict(rechunk_args)
    rechunk_args["target_chunks"] = (10, 10)
    with pytest.raises(
        ValueError, match="You must specify ``target-chunks`` as a dict"
    ):
        api.rechunk(**rechunk_args)


def test_rechunk_invalid_source(tmp_path):
    with pytest.raises(
        ValueError,
        match="Source must be a Zarr Array, Zarr Group, Dask Array or Xarray Dataset",
    ):
        api.rechunk(
            [[1, 2], [3, 4]], target_chunks=(10, 10), max_mem=100, target_store=tmp_path
        )


@pytest.mark.parametrize(
    "source,target_chunks",
    [
        (sample_xarray_dataset(), {"a": (10, 5, 4), "b": (100,)}),
        (dsa.ones((20, 10), chunks=(5, 5)), (10, 10)),
    ],
)
@pytest.mark.parametrize(
    "executor",
    [
        "python",
        requires_beam("beam"),
        requires_prefect("prefect"),
        requires_pywren("pywren"),
    ],
)
def test_unsupported_executor(tmp_path, source, target_chunks, executor):
    with pytest.raises(
        NotImplementedError, match="Executor type .* not supported for source",
    ):
        api.rechunk(
            source,
            target_chunks=target_chunks,
            max_mem=1600,
            target_store=str(tmp_path / "target.zarr"),
            temp_store=str(tmp_path / "temp.zarr"),
            executor=executor,
        )


def test_rechunk_no_target_chunks(rechunk_args):
    rechunk_args = dict(rechunk_args)
    if _is_collection(rechunk_args["source"]):
        rechunk_args["target_chunks"] = {v: None for v in rechunk_args["source"]}
    else:
        rechunk_args["target_chunks"] = None
    api.rechunk(**rechunk_args)


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
