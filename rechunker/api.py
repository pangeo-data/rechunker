"""User-facing functions."""
import html
import textwrap
import uuid

import zarr
import dask
import dask.array as dsa
from dask.optimization import fuse
from dask.delayed import Delayed


from rechunker.algorithm import rechunking_plan


class Rechunked(Delayed):
    """
    A delayed rechunked result.

    This represents the rechunking plan, and when executed will perform
    the rechunking and return the rechunked array.

    Methods
    -------

    Examples
    --------
    >>> source = zarr.ones((4, 4), chunks=(2, 2), store="source.zarr")
    >>> intermediate = "intermediate.zarr"
    >>> target = "target.zarr"
    >>> rechunked = rechunk(source, target_chunks=(4, 1), target_store=target,
    ...                     max_mem=256000,
    ...                     temp_store=intermediate)
    >>> rechunked
    <Rechunked>
    * Source      : <zarr.core.Array (4, 4) float64>
    * Intermediate: dask.array<from-zarr, ... >
    * Target      : <zarr.core.Array (4, 4) float64>
    >>> rechunked.execute()
    <zarr.core.Array (4, 4) float64>
    """

    __slots__ = (
        "_key",
        "dask",
        "_length",
        "_source",
        "_intermediate",
        "_target",
        "_target_original",
    )

    def __init__(
        self,
        key,
        dsk,
        length=None,
        *,
        source,
        intermediate,
        target,
        target_original=None,
    ):
        self._source = source
        self._intermediate = intermediate
        self._target = target
        self._target_original = target_original
        super().__init__(key, dsk, length=length)

    def execute(self, **kwargs):
        """
        Execute the rechunking.

        Parameters
        ----------
        scheduler : string, optional
            Which Dask scheduler to use like "threads", "synchronous" or "processes".
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
        if self._target_original is None:
            return self._target

        target = zarr.open(self._target_original)
        target.append(self._target)
        return target

    def __repr__(self):
        if self._intermediate is not None:
            intermediate = f"\n* Intermediate: {repr(self._intermediate)}"
        else:
            intermediate = ""

        return textwrap.dedent(
            f"""\
            <Rechunked>
            * Source      : {repr(self._source)}{{}}
            * Target      : {repr(self._target)}
            """
        ).format(intermediate)

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
          {{source_html}}
        </details>
        {}
        <details>
          <summary><b>Target</b></summary>
          {{target_html}}
        </details>
        """
        )

        if self._intermediate is not None:
            intermediate = textwrap.dedent(
                """\
                <details>
                <summary><b>Intermediate</b></summary>
                {intermediate_html}
                </details>
            """
            )
        else:
            intermediate = ""
        template = template.format(intermediate)
        return template.format(**entries)


def _shape_dict_to_tuple(dims, shape_dict):
    # convert a dict of shape
    shape = [shape_dict[dim] for dim in dims]
    return tuple(shape)


def _get_dims_from_zarr_array(z_array):
    # use Xarray convention
    # http://xarray.pydata.org/en/stable/internals.html#zarr-encoding-specification
    return z_array.attrs["_ARRAY_DIMENSIONS"]


def _zarr_empty(shape, store_or_group, chunks, dtype, name=None):
    # wrapper that maybe creates the array within a group
    if name is not None:
        assert isinstance(store_or_group, zarr.hierarchy.Group)
        return store_or_group.empty(name, shape=shape, chunks=chunks, dtype=dtype)
    else:
        return zarr.empty(shape, chunks=chunks, dtype=dtype, store=store_or_group)


def _reduce(*args):
    return None


def _barrier(*args):
    return None


def _result(result, *args):
    return result


def rechunk(
    source,
    target_chunks,
    max_mem,
    target_store,
    temp_store=None,
    source_slice=None,
    target_append=False,
):
    """
    Rechunk a Zarr Array or Group

    Parameters
    ----------
    source : zarr.Array or zarr.Group
        Named dimensions in the Arrays will be parsed according to the
        Xarray :ref:`xarray:zarr_encoding`.
    target_chunks : tuple, dict, or None
        The desired chunks of the array after rechunking. The structure
        depends on ``source``.

        - For a single array source, ``target_chunks`` can
          be either a tuple (e.g. ``(20, 5, 3)``) or a dictionary
          (e.g. ``{'time': 20, 'lat': 5, 'lon': 3}``). Dictionary syntax
          requires the dimension names be present in the Zarr Array
          attributes (see Xarray :ref:`xarray:zarr_encoding`.)
          A value of ``None`` means that the array will
          be copied with no change to its chunk structure.
        - For a group, a dict is required. The keys correspond to array names.
          The values are ``target_chunks`` arguments for the array. For example,
          ``{'foo': (20, 10), 'bar': {'x': 3, 'y': 5}, 'baz': None}``.
          *All arrays you want to rechunk must be explicitly named.* Arrays
          that are not present in the ``target_chunks`` dict will be ignored.

    max_mem : str or int
        The amount of memory (in bytes) that workers are allowed to use. A
        string (e.g. ``100MB``) can also be used.
    target_store : str, MutableMapping, or zarr.Store object
        The location in which to store the final, rechunked result.
        Will be passed directly to :py:meth:`zarr.creation.create`
    temp_store : str, MutableMapping, or zarr.Store object, optional
        Location of temporary store for intermediate data. Can be deleted
        once rechunking is complete.
    source_slice : tuple, dict or None
        The slice of the source to rechunk. The structure depends on ``source``.

        - For a single array source, ``source_slice`` can be either a tuple (e.g.
        ``((0, 20), None, None)`` or a dictionary (e.g. ``{'time': (0, 20),
        'lat': None, 'lon': None}``). Dictionary syntax requires the dimension
        names be present in the Zarr Array attributes (see Xarray :ref:`xarray:zarr_encoding`.)
        A value of ``None`` means that the whole source array will be rechunked.
        - For a group, a dict is required. The keys correspond to array names.
        The values are ``source_slice`` arguments for the array. For example,
        ``{'foo': ((0, 20), None, None), 'bar': {'time': (0, 20),
        'lat': None, 'lon': None}, 'baz': None}``.
        *All arrays you want to slice must be explicitly named.* Arrays
        that are not present in the ``source_slice`` dict will be ignored.
    target_append : bool, optional
        Whether to append the rechunked result to ``target_store`` or not. If ``True``,
        ``target_store`` must alreay exist.

    Returns
    -------
    rechunked : :class:`Rechunked` object
    """

    # these options are not tested yet; don't include in public API
    kwargs = dict(
        source_storage_options={}, temp_storage_options={}, target_storage_options={},
    )

    if isinstance(source, zarr.hierarchy.Group):
        if not isinstance(target_chunks, dict):
            raise ValueError(
                "You must specificy ``target-chunks`` as a dict when rechunking a group."
            )

        stores_delayed = []

        if temp_store:
            temp_group = zarr.group(temp_store)
        else:
            temp_group = None
        target_group = zarr.group(target_store)
        target_group.attrs.update(source.attrs)

        for array_name, array_target_chunks in target_chunks.items():
            delayed = _rechunk_array(
                source[array_name],
                array_target_chunks,
                max_mem,
                target_group,
                temp_store_or_group=temp_group,
                name=array_name,
                **kwargs,
            )
            stores_delayed.append(delayed)

        # This next block makes a task that
        # 1. Returns the target Group (see dsk[name] = ...)...
        # 2. which depends on each of the component arrays
        # 3. but doesn't require transmitting large dependencies (depend on barrier_name,
        #    rather than on part.key directly) to compute the result
        always_new_token = uuid.uuid1().hex
        barrier_name = "barrier-" + always_new_token
        dsk2 = {
            (barrier_name, i): (_barrier, part.key)
            for i, part in enumerate(stores_delayed)
        }

        name = "rechunked-" + dask.base.tokenize([x.name for x in stores_delayed])
        dsk = dask.base.merge(*[x.dask for x in stores_delayed], dsk2)
        dsk[name] = (_result, target_group,) + tuple(
            (barrier_name, i) for i, _ in enumerate(stores_delayed)
        )
        rechunked = Rechunked(
            name, dsk, source=source, intermediate=temp_group, target=target_group,
        )

        return rechunked

    elif isinstance(source, zarr.core.Array):
        return _rechunk_array(
            source,
            target_chunks,
            max_mem,
            target_store,
            temp_store_or_group=temp_store,
            source_slice=source_slice,
            target_append=target_append,
            **kwargs,
        )

    else:
        raise ValueError("Source must be a Zarr Array or Group.")


def _rechunk_array(
    source_array,
    target_chunks,
    max_mem,
    target_store_or_group,
    temp_store_or_group=None,
    source_slice=None,
    target_append=False,
    name=None,
    source_storage_options={},
    temp_storage_options={},
    target_storage_options={},
):

    if source_slice is None:
        shape = source_array.shape
    else:
        source_slice = tuple(
            slice(None) if s is None else slice(*s) for s in source_slice
        )
        shape = tuple(
            len(range(*sl.indices(sh)))
            for sh, sl in zip(source_array.shape, source_slice)
        )
    source_chunks = source_array.chunks
    dtype = source_array.dtype
    itemsize = dtype.itemsize

    if target_chunks is None:
        # this is just a pass-through copy
        target_chunks = source_chunks

    if isinstance(target_chunks, dict):
        array_dims = _get_dims_from_zarr_array(source_array)
        try:
            target_chunks = _shape_dict_to_tuple(array_dims, target_chunks)
        except KeyError:
            raise KeyError(
                "You must explicitly specify each dimension size in target_chunks. "
                f"Got array_dims {array_dims}, target_chunks {target_chunks}."
            )

    max_mem = dask.utils.parse_bytes(max_mem)

    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape, source_chunks, target_chunks, itemsize, max_mem
    )

    source_read = dsa.from_zarr(
        source_array, chunks=read_chunks, storage_options=source_storage_options
    )
    if source_slice is not None:
        source_read = source_read[source_slice]

    # create target
    shape = tuple(int(x) for x in shape)  # ensure python ints for serialization
    target_chunks = tuple(int(x) for x in target_chunks)
    int_chunks = tuple(int(x) for x in int_chunks)
    write_chunks = tuple(int(x) for x in write_chunks)

    if target_append:
        target_store_or_group_original = target_store_or_group
        target_store_or_group = "to_append_" + target_store_or_group
    else:
        target_store_or_group_original = None

    target_array = _zarr_empty(
        shape, target_store_or_group, target_chunks, dtype, name=name
    )
    target_array.attrs.update(source_array.attrs)

    if read_chunks == write_chunks:
        target_store_delayed = dsa.store(
            source_read, target_array, lock=False, compute=False
        )

        # fuse
        target_dsk = dask.utils.ensure_dict(target_store_delayed.dask)
        dsk_fused, deps = fuse(target_dsk)

        return Rechunked(
            target_store_delayed.key,
            dsk_fused,
            source=source_array,
            intermediate=None,
            target=target_array,
            target_original=target_store_or_group_original,
        )

    else:
        # do intermediate store
        assert temp_store_or_group is not None
        int_array = _zarr_empty(
            shape, temp_store_or_group, int_chunks, dtype, name=name
        )
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
            target_original=target_store_or_group_original,
        )

        return delayed_fused
