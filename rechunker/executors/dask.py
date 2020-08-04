import uuid
from typing import Iterable, Tuple

import dask
import dask.array
from dask.delayed import Delayed
from dask.optimization import fuse

from rechunker.types import Array, CopySpec, Executor


class DaskExecutor(Executor[Delayed]):
    """An execution engine based on dask.

    Supports zarr and dask arrays as inputs. Outputs must be zarr arrays.

    Execution plans for DaskExecutors are dask.delayed objects.
    """

    def prepare_plan(self, specs: Iterable[CopySpec]) -> Delayed:
        return _copy_all(specs)

    def execute_plan(self, plan: Delayed, **kwargs):
        return plan.compute(**kwargs)


def _direct_array_copy(
    source: Array, target: Array, chunks: Tuple[int, ...]
) -> Delayed:
    """Direct copy between arrays."""
    if isinstance(source, dask.array.Array):
        source_read = source
    else:
        source_read = dask.array.from_zarr(source, chunks=chunks)
    target_store_delayed = dask.array.store(
        source_read, target, lock=False, compute=False
    )
    return target_store_delayed


def _chunked_array_copy(spec: CopySpec) -> Delayed:
    """Chunked copy between arrays."""
    if spec.intermediate.array is None:
        target_store_delayed = _direct_array_copy(
            spec.read.array, spec.write.array, spec.read.chunks,
        )

        # fuse
        target_dsk = dask.utils.ensure_dict(target_store_delayed.dask)
        dsk_fused, _ = fuse(target_dsk)

        return Delayed(target_store_delayed.key, dsk_fused)

    else:
        # do intermediate store
        int_store_delayed = _direct_array_copy(
            spec.read.array, spec.intermediate.array, spec.read.chunks,
        )
        target_store_delayed = _direct_array_copy(
            spec.intermediate.array, spec.write.array, spec.write.chunks,
        )

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


def _barrier(*args):
    return None


def _copy_all(specs: Iterable[CopySpec],) -> Delayed:

    stores_delayed = [_chunked_array_copy(spec) for spec in specs]

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
