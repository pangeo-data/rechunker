"""User-facing functions."""

import zarr
import dask
import dask.array as dsa
from dask.optimization import fuse
from dask.delayed import Delayed


from rechunker.algorithm import rechunking_plan


def _shape_dict_to_tuple(dims, shape_dict):
    # convert a dict of shape
    shape = [shape_dict[dim] for dim in dims]
    return tuple(shape)


def _get_dims_from_zarr_array(z_array):
    # use Xarray convention
    # http://xarray.pydata.org/en/stable/internals.html#zarr-encoding-specification
    return z_array.attrs['_ARRAY_DIMENSIONS']


def rechunk(
    source_array,
    target_chunks,
    max_mem,
    target_store,
    temp_store=None,
    source_storage_options={},
    temp_storage_options={},
    target_storage_options={},
):
    return _rechunk_zarr2zarr_w_dask(
        source_array,
        target_chunks,
        max_mem,
        target_store,
        temp_store=temp_store,
        source_storage_options=source_storage_options,
        temp_storage_options=temp_storage_options,
        target_storage_options=target_storage_options,
    )


def _rechunk_zarr2zarr_w_dask(
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

    if isinstance(target_chunks, dict):
        array_dims = _get_dims_from_zarr_array(source_array)
        try:
            target_chunks = _shape_dict_to_tuple(array_dims, target_chunks)
        except KeyError:
            raise KeyError("You must explicitly specify each dimension size in target_chunks. "
                           f"Got array_dims {array_dims}, target_chunks {target_chunks}.")

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )

    print(source_chunks, read_chunks, int_chunks, write_chunks, target_chunks)

    source_read = dsa.from_zarr(
        source_array, chunks=read_chunks, storage_options=source_storage_options
    )

    # create target
    shape = tuple(int(x) for x in shape)  # ensure python ints for serialization
    target_chunks = tuple(int(x) for x in target_chunks)
    int_chunks = tuple(int(x) for x in int_chunks)
    write_chunks = tuple(int(x) for x in write_chunks)

    target_array = zarr.empty(
        shape, chunks=target_chunks, dtype=dtype, store=target_store
    )
    target_array.attrs.update(source_array.attrs)

    if read_chunks == write_chunks:
        target_store_delayed = dsa.store(
            source_read, target_array, lock=False, compute=False
        )
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
