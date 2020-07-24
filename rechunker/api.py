"""User-facing functions."""
import html
import textwrap

import zarr
import dask

from rechunker.algorithm import rechunking_plan
import rechunker.dask
from rechunker.types import CopySpec, StagedCopySpec


class Rechunked:
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

    def __init__(self, delayed, source, intermediate, target):
        self._delayed = delayed
        self._source = source
        self._intermediate = intermediate
        self._target = target

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
        self._delayed.compute(**kwargs)
        return self._target

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


def rechunk(
    source,
    target_chunks,
    max_mem,
    target_store,
    temp_store=None,
    staged_copy=rechunker.dask.staged_copy,
):
    """
    Rechunk a Zarr Array or Group, or a Dask Array

    Parameters
    ----------
    source : zarr.Array, zarr.Group, or dask.array.Array
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
    staged_copy: function
        Implementation of copying between zarr arrays.

    Returns
    -------
    rechunked : :class:`Rechunked` object
    """
    copy_specs, intermediate, target = _setup_rechunk(
        source, target_chunks, max_mem, target_store, temp_store
    )
    delayed = staged_copy(copy_specs)
    return Rechunked(delayed, source, intermediate, target)


def _setup_rechunk(
    source, target_chunks, max_mem, target_store, temp_store=None,
):
    if isinstance(source, zarr.hierarchy.Group):
        if not isinstance(target_chunks, dict):
            raise ValueError(
                "You must specify ``target-chunks`` as a dict when rechunking a group."
            )

        if temp_store:
            temp_group = zarr.group(temp_store)
        else:
            temp_group = None
        target_group = zarr.group(target_store)
        target_group.attrs.update(source.attrs)

        copy_specs = []
        for array_name, array_target_chunks in target_chunks.items():
            copy_spec = _setup_array_rechunk(
                source[array_name],
                array_target_chunks,
                max_mem,
                target_group,
                temp_store_or_group=temp_group,
                name=array_name,
            )
            copy_specs.append(copy_spec)

        return copy_specs, temp_group, target_group

    elif isinstance(source, zarr.core.Array) or isinstance(source, dask.array.Array):

        copy_spec = _setup_array_rechunk(
            source,
            target_chunks,
            max_mem,
            target_store,
            temp_store_or_group=temp_store,
        )
        intermediate = (
            copy_spec.stages[0].target if len(copy_spec.stages) == 2 else None
        )
        target = copy_spec.stages[-1].target
        return [copy_spec], intermediate, target

    else:
        raise ValueError("Source must be a Zarr Array or Group, or a Dask Array.")


def _setup_array_rechunk(
    source_array,
    target_chunks,
    max_mem,
    target_store_or_group,
    temp_store_or_group=None,
    name=None,
) -> StagedCopySpec:
    shape = source_array.shape
    source_chunks = (
        source_array.chunksize
        if isinstance(source_array, dask.array.Array)
        else source_array.chunks
    )
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

    # TODO: rewrite to avoid the hard dependency on dask
    max_mem = dask.utils.parse_bytes(max_mem)

    # don't consolidate reads for Dask arrays
    consolidate_reads = isinstance(source_array, zarr.core.Array)
    read_chunks, int_chunks, write_chunks = rechunking_plan(
        shape,
        source_chunks,
        target_chunks,
        itemsize,
        max_mem,
        consolidate_reads=consolidate_reads,
    )

    # create target
    shape = tuple(int(x) for x in shape)  # ensure python ints for serialization
    target_chunks = tuple(int(x) for x in target_chunks)
    int_chunks = tuple(int(x) for x in int_chunks)
    write_chunks = tuple(int(x) for x in write_chunks)

    target_array = _zarr_empty(
        shape, target_store_or_group, target_chunks, dtype, name=name
    )
    try:
        target_array.attrs.update(source_array.attrs)
    except AttributeError:
        pass

    if read_chunks == write_chunks:
        return StagedCopySpec([CopySpec(source_array, target_array, read_chunks)])
    else:
        # do intermediate store
        assert temp_store_or_group is not None
        int_array = _zarr_empty(
            shape, temp_store_or_group, int_chunks, dtype, name=name
        )
        return StagedCopySpec(
            [
                CopySpec(source_array, int_array, read_chunks),
                CopySpec(int_array, target_array, write_chunks),
            ]
        )
