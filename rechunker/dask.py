import uuid
from typing import Iterable

import dask
import dask.array
from dask.delayed import Delayed
from dask.optimization import fuse

from rechunker.core import CopySpec, StagedCopySpec


def _direct_copy_array(copy_spec: CopySpec) -> Delayed:
    """Direct copy between zarr arrays."""
    source_array, target_array, chunks = copy_spec
    if isinstance(source_array, dask.array.Array):
        source_read = source_array
    else:
        source_read = dask.array.from_zarr(source_array, chunks=chunks)
    target_store_delayed = dask.array.store(
        source_read, target_array, lock=False, compute=False
    )
    return target_store_delayed


def _staged_array_copy(staged_copy_spec: StagedCopySpec) -> Delayed:
    """Staged copy between zarr arrays."""
    if len(staged_copy_spec.stages) == 1:
        (copy_spec,) = staged_copy_spec.stages
        target_store_delayed = _direct_copy_array(copy_spec)

        # fuse
        target_dsk = dask.utils.ensure_dict(target_store_delayed.dask)
        dsk_fused, _ = fuse(target_dsk)

        return Delayed(target_store_delayed.key, dsk_fused)

    elif len(staged_copy_spec.stages) == 2:
        first_copy, second_copy = staged_copy_spec.stages

        # do intermediate store
        int_store_delayed = _direct_copy_array(first_copy)
        target_store_delayed = _direct_copy_array(second_copy)

        # now do some hacking to chain these together into a single graph.
        # get the two graphs as dicts
        int_dsk = dask.utils.ensure_dict(int_store_delayed.dask)
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
            *int_dsk[int_store_delayed.key],
        )
        target_dsk.update(int_dsk)

        # fuse
        dsk_fused, _ = fuse(target_dsk)
        return Delayed(target_store_delayed.key, dsk_fused)
    else:
        raise NotImplementedError


def _barrier(*args):
    return None


def staged_copy(
    staged_copy_specs: Iterable[StagedCopySpec],
) -> Delayed:

    stores_delayed = [_staged_array_copy(spec) for spec in staged_copy_specs]

    if len(stores_delayed) == 1:
        return stores_delayed[0]

    # This next block makes a task that
    # 1. depends on each of the component arrays
    # 2. but doesn't require transmitting large dependencies (depend on barrier_name,
    #    rather than on part.key directly) to compute the result
    always_new_token = uuid.uuid1().hex
    barrier_name = "barrier-" + always_new_token
    dsk2 = {
        (barrier_name, i): (_barrier, part.key) for i, part in enumerate(stores_delayed)
    }

    name = "rechunked-" + dask.base.tokenize([x.name for x in stores_delayed])
    dsk = dask.base.merge(*[x.dask for x in stores_delayed], dsk2)
    dsk[name] = (_barrier,) + tuple(
        (barrier_name, i) for i, _ in enumerate(stores_delayed)
    )
    return Delayed(name, dsk)
