import dask
import dask.core
import numpy as np
import pytest
import xarray
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from rechunker import api
from unittest.mock import MagicMock, patch

TEST_DATASET = xarray.DataArray(
    data=np.empty((10, 10)),
    coords={"x": range(0, 10), "y": range(0, 10)},
    dims=["x", "y"],
    name="test_data",
).to_dataset()
LOCAL_FS = LocalFileSystem()
Mem = MemoryFileSystem()
TARGET_STORE_NAME = "target_store.zarr"
TMP_STORE_NAME = "tmp.zarr"


class Test_rechunk_cm:
    def _clean(self, stores):
        for s in stores:
            try:
                LOCAL_FS.rm(s, recursive=True, maxdepth=100)
            except:
                pass

    @pytest.fixture(autouse=True)
    def _wrap(self):
        self._clean([TMP_STORE_NAME, TARGET_STORE_NAME])
        with dask.config.set(scheduler="single-threaded"):
            yield
        self._clean([TMP_STORE_NAME, TARGET_STORE_NAME])

    @patch("rechunker.api.rechunk")
    def test_rechunk_cm__args_sent_as_is(self, rechunk_func: MagicMock):
        with api.rechunk_cm(
            source="source",
            target_chunks={"truc": "bidule"},
            max_mem="42KB",
            target_store="target_store.zarr",
            temp_store="tmp_store.zarr",
            target_options=None,
            temp_options=None,
            executor="dask",
            filesystem=LOCAL_FS,
            keep_target_store=False,
        ):
            rechunk_func.assert_called_with(
                source="source",
                target_chunks={"truc": "bidule"},
                max_mem="42KB",
                target_store="target_store.zarr",
                target_options=None,
                temp_store="tmp_store.zarr",
                temp_options=None,
                executor="dask",
            )

    def test_rechunk_cm__remove_every_stores(self):
        with api.rechunk_cm(
            source=TEST_DATASET,
            target_chunks={"x": 2, "y": 2},
            max_mem="42KB",
            target_store="target_store.zarr",
            temp_store="tmp_store.zarr",
            target_options=None,
            temp_options=None,
            executor="dask",
            filesystem=LOCAL_FS,
            keep_target_store=False,
        ) as plan:
            plan.execute()
            assert LOCAL_FS.exists("target_store.zarr")
            assert LOCAL_FS.exists("tmp_store.zarr")
        assert not LOCAL_FS.exists("tmp_store.zarr")
        assert not LOCAL_FS.exists("target_store.zarr")

    def test_rechunk_cm__keep_target(self):
        with api.rechunk_cm(
            source=TEST_DATASET,
            target_chunks={"x": 2, "y": 2},
            max_mem="42KB",
            target_store="target_store.zarr",
            temp_store="tmp_store.zarr",
            target_options=None,
            temp_options=None,
            executor="dask",
            filesystem=LOCAL_FS,
            keep_target_store=True,
        ) as plan:
            plan.execute()
            assert LOCAL_FS.exists("target_store.zarr")
            assert LOCAL_FS.exists("tmp_store.zarr")
        assert LOCAL_FS.exists("target_store.zarr")
        assert not LOCAL_FS.exists("tmp_store.zarr")

    def test_rechunk_cm__error_target_exist(self):
        f = LOCAL_FS.open("target_store.zarr", "x")
        f.close()
        with pytest.raises(FileExistsError):
            with api.rechunk_cm(
                source=TEST_DATASET,
                target_chunks={"x": 2, "y": 2},
                max_mem="42KB",
                target_store="target_store.zarr",
                temp_store="tmp_store.zarr",
                target_options=None,
                temp_options=None,
                executor="dask",
                filesystem=LOCAL_FS,
                keep_target_store=False,
            ):
                pass
