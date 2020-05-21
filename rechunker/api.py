"""User-facing functions."""

import zarr
import dask
import dask.array as dsa
from dask.optimization import fuse
from dask.delayed import Delayed


from rechunker.algorithm import rechunking_plan


def rechunk_zarr2zarr_w_dask(
    source_array,
    target_chunks,
    max_mem,
    target_store,
    temp_store=None,
    source_storage_options={},
    temp_storage_options={},
    target_storage_options={},
):

    shape = source_array.shape
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )

    source_read = dsa.from_zarr(
        source_array, chunks=read_chunks, storage_options=source_storage_options
    )

    # create target
    target_array = zarr.empty(
        shape, chunks=target_chunks, dtype=dtype, store=target_store
    )
    target_array.attrs.update(source_array.attrs)

    if int_chunks == target_chunks:
        target_store_delayed = dsa.store(
            source_read, target_array, lock=False, compute=False
        )
        print("One step rechunking plan")
        return target_store_delayed

    else:
        # do intermediate store
        assert temp_store is not None
        int_array = zarr.empty(shape, chunks=int_chunks, dtype=dtype, store=temp_store)
        intermediate_store_delayed = dsa.store(
            source_read, int_array, lock=False, compute=False
        )

        int_read = dsa.from_zarr(
            int_array, chunks=write_chunks, storage_options=temp_storage_options
        )
        target_store_delayed = dsa.store(
            int_read, target_array, lock=False, compute=False
        )

        # now do some hacking to chain these together into a single graph.
        # get the two graphs as dicts
        int_dsk = dask.utils.ensure_dict(intermediate_store_delayed.dask)
        target_dsk = dask.utils.ensure_dict(target_store_delayed.dask)

        # find the root store key representing the read
        root_keys = []
        for key in target_dsk:
            if isinstance(key, str):
                if key.startswith("from-zarr"):
                    root_keys.append(key)
        assert len(root_keys) == 1
        root_key = root_keys[0]

        # now rewrite the graph
        target_dsk[root_key] = (
            lambda a, *b: a,
            target_dsk[root_key],
            *int_dsk[intermediate_store_delayed.key],
        )
        target_dsk.update(int_dsk)

        # fuse
        dsk_fused, deps = fuse(target_dsk)
        delayed_fused = Delayed(target_store_delayed.key, dsk_fused)

        print("Two step rechunking plan")
        return delayed_fused
