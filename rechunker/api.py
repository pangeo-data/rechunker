"""User-facing functions."""
import html
import textwrap

import zarr
import dask
import dask.array as dsa
from dask.optimization import fuse
from dask.delayed import Delayed


from rechunker.algorithm import rechunking_plan


class Rechunked(Delayed):
    __slots__ = ("_key", "dask", "_length", "_source", "_intermediate", "_target")

    def __init__(self, key, dsk, length=None, *, source, intermediate, target):
        self._source = source
        self._intermediate = intermediate
        self._target = target
        super().__init__(key, dsk, length=length)

    def execute(self, **kwargs):
        """
        Execute the rechunking.

        Parameters
        ----------
        scheduler : string, optional
            Which scheduler to use like "threads", "synchronous" or "processes".
            If not provided, the default is to check the global settings first,
            and then fall back to the collection defaults.
        optimize_graph : bool, optional
            If True [default], the graph is optimized before computation.
            Otherwise the graph is run as is. This can be useful for debugging.
        kwargs
            Extra keywords to forward to the scheduler function.

        Returns
        -------
        The same type of the ``source_array`` originally provided to
        :func:`rechunker.rechunk`.
        """
        self.compute(**kwargs)
        return self._target

    def __repr__(self):
        return textwrap.dedent(
            f"""\
            <Rechunked>
            * Source      : {repr(self._source)}
            * Intermediate: {repr(self._intermediate)}
            * Target      : {repr(self._target)}
            """
        )

    def _repr_html_(self):
        entries = {}
        for kind, obj in [
            ("source", self._source),
            ("intermediate", self._intermediate),
            ("target", self._target),
        ]:
            try:
                body = obj._repr_html_()
            except AttributeError:
                body = f"<p><code>{html.escape(repr(self._target))}</code></p>"
            entries[f"{kind}_html"] = body

        template = textwrap.dedent(
            """<h2>Rechunked</h2>\

        <details>
          <summary><b>Source</b></summary>
          {source_html}
        </details>

        <details>
          <summary><b>Intermediate</b></summary>
          {intermediate_html}
        </details>

        <details>
          <summary><b>Target</b></summary>
          {target_html}
        </details>
        """
        )
        return template.format(**entries)


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
            int_read, target_array, lock=False, compute=False,
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
        delayed_fused = Rechunked(
            target_store_delayed.key,
            dsk_fused,
            source=source_array,
            intermediate=int_read,
            target=target_array,
        )

        return delayed_fused
