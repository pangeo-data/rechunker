import pytest

import zarr
import dask.array as dsa

from rechunker import api


@pytest.fixture
def rechunk_delayed(tmp_path):
    store_source = str(tmp_path / "source.zarr")
    shape = (8000, 8000)
    source_chunks = (200, 8000)
    dtype = "f4"

    a_source = zarr.ones(shape, chunks=source_chunks, dtype=dtype, store=store_source)

    target_store = str(tmp_path / "target.zarr")
    temp_store = str(tmp_path / "temp.zarr")
    max_mem = 25600000
    target_chunks = (8000, 200)
    return (
        api.rechunk_zarr2zarr_w_dask(
            a_source, target_chunks, max_mem, target_store, temp_store=temp_store
        ),
        target_store,
    )


def test_compute(rechunk_delayed):
    rechunked, target_store = rechunk_delayed
    result = rechunked.execute()
    assert isinstance(result, zarr.Array)
    a_tar = dsa.from_zarr(result)
    assert dsa.equal(a_tar, 1).all().compute()


def test_repr(rechunk_delayed):
    rechunked, target_store = rechunk_delayed
    assert isinstance(rechunked, api.Rechunked)
    repr_str = repr(rechunked)

    assert repr_str.startswith("<Rechunked>")
    assert all(thing in repr_str for thing in ["Source", "Intermediate", "Target"])


def test_rerp_html(rechunk_delayed):
    rechunked, target_store = rechunk_delayed
    rechunked._repr_html_()  # no exceptions
