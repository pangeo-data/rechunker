"""User-facing functions."""

import zarr
import dask
import dask.array as dsa

from rechunker.algorithm import rechunking_plan

def rechunk_zarr2zarr_w_dask(source_array, target_chunks, max_mem,
                             target_store, temp_store=None,
                             source_storage_options={},
                             temp_storage_options={},
                             target_storage_options={}):

    shape = source_array.shape
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )


    array_read = dsa.from_zarr(source_array, chunks=source_chunks,
                               storage_options=source_storage_options)

    store_futures = []

    # create target
    target_array = zarr.empty(shape, chunks=target_chunks, dtype=dtype, store=target_store)
    target_array.attrs.update(source_array.attrs)


    if int_chunks == target_chunks:
        store_future = dsa.store(array_read, target_array, lock=False, compute=False)
        store_futures.append(store_future)
    else:
        # do intermediate chunks
