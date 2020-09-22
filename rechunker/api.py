"""User-facing functions."""
import html
import textwrap
from typing import Union, Mapping

import zarr
import dask
import dask.array
import xarray
import tempfile

from rechunker.algorithm import rechunking_plan
from rechunker.types import ArrayProxy, CopySpec, Executor
from xarray.backends.zarr import (
    encode_zarr_attr_value,
    encode_zarr_variable,
    extract_zarr_variable_encoding,
    DIMENSION_KEY,
)


class Rechunked:
    """
    A delayed rechunked result.

    This represents the rechunking plan, and when executed will perform
    the rechunking and return the rechunked array.

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

    def __init__(self, executor, plan, source, intermediate, target):
        self._executor = executor
        self._plan = plan
        self._source = source
        self._intermediate = intermediate
        self._target = target

    @property
    def plan(self):
        """Returns the executor-specific scheduling plan.

        The type of this object depends on the underlying execution engine.
        """
        return self._plan

    def execute(self, **kwargs):
        """
        Execute the rechunking.

        Parameters
        ----------
        **kwargs
            Keyword arguments are forwarded to the executor's ``execute_plan``
            method.

        Returns
        -------
        The same type of the ``source_array`` originally provided to
        :func:`rechunker.rechunk`.
        """
        self._executor.execute_plan(self._plan, **kwargs)
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


def _zarr_empty(shape, store_or_group, chunks, dtype, name=None, **kwargs):
    # wrapper that maybe creates the array within a group
    if name is not None:
        assert isinstance(store_or_group, zarr.hierarchy.Group)
        return store_or_group.empty(
            name, shape=shape, chunks=chunks, dtype=dtype, **kwargs
        )
    else:
        return zarr.empty(
            shape, chunks=chunks, dtype=dtype, store=store_or_group, **kwargs
        )


def _get_executor(name: str) -> Executor:
    # converts a string name into a Executor instance
    # imports are conditional to avoid hard dependencies
    if name.lower() == "dask":
        from rechunker.executors.dask import DaskExecutor

        return DaskExecutor()
    elif name.lower() == "beam":
        from rechunker.executors.beam import BeamExecutor

        return BeamExecutor()
    elif name.lower() == "prefect":
        from rechunker.executors.prefect import PrefectExecutor

        return PrefectExecutor()
    elif name.lower() == "python":
        from rechunker.executors.python import PythonExecutor

        return PythonExecutor()
    elif name.lower() == "pywren":
        from rechunker.executors.pywren import PywrenExecutor

        return PywrenExecutor()
    else:
        raise ValueError(f"unrecognized executor {name}")


def rechunk_dataset(
    source: xarray.Dataset,
    encoding: Mapping,
    max_mem,
    target_store,
    temp_store=None,
    executor: Union[str, Executor] = "dask",
):
    def _encode_zarr_attributes(attrs):
        return {k: encode_zarr_attr_value(v) for k, v in attrs.items()}

    if isinstance(executor, str):
        executor = _get_executor(executor)
    if temp_store:
        temp_group = zarr.group(temp_store)
    else:
        temp_group = zarr.group(tempfile.mkdtemp(".zarr", "temp_store_"))
    target_group = zarr.group(target_store)
    target_group.attrs.update(_encode_zarr_attributes(source.attrs))

    copy_specs = []
    for variable in source:
        array = source[variable].copy()

        # Update the array encoding with provided parameters and apply it
        has_chunk_encoding = "chunks" in array.encoding
        array.encoding.update(encoding.get(variable, {}))
        array = encode_zarr_variable(array)

        # Determine target chunking for array and remove it prior to
        # validation/extraction ONLY if the array isn't also coming
        # from a Zarr store (otherwise blocks need to be checked for overlap)
        target_chunks = array.encoding.get("chunks")
        if not has_chunk_encoding:
            array.encoding.pop("chunks", None)
        array_encoding = extract_zarr_variable_encoding(
            array, raise_on_invalid=True, name=variable
        )

        # Default to chunking based on array shape if not explicitly provided
        default_chunks = array_encoding.pop("chunks")
        target_chunks = target_chunks or default_chunks

        # Extract array attributes along with reserved property for
        # xarray dimension names
        array_attrs = _encode_zarr_attributes(array.attrs)
        array_attrs[DIMENSION_KEY] = encode_zarr_attr_value(array.dims)

        copy_spec = _setup_array_rechunk(
            dask.array.asarray(array),
            target_chunks,
            max_mem,
            target_group,
            target_options=array_encoding,
            temp_store_or_group=temp_group,
            temp_options=array_encoding,
            name=variable,
        )
        copy_spec.write.array.attrs.update(array_attrs)  # type: ignore
        copy_specs.append(copy_spec)
    plan = executor.prepare_plan(copy_specs)
    return Rechunked(executor, plan, source, temp_group, target_group)


def rechunk(
    source,
    target_chunks,
    max_mem,
    target_store,
    target_options=None,
    temp_store=None,
    temp_options=None,
    executor: Union[str, Executor] = "dask",
) -> Rechunked:
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
    target_options: Dict, optional
        Additional keyword arguments used to create target arrays.
        See :py:meth:`zarr.creation.create` for arguments available.
        Must not include any of [``shape``, ``chunks``, ``dtype``, ``store``].
    temp_store : str, MutableMapping, or zarr.Store object, optional
        Location of temporary store for intermediate data. Can be deleted
        once rechunking is complete.
    temp_options: Dict, optional
        Additional keyword arguments used to create intermediate arrays.
        See :py:meth:`zarr.creation.create` for arguments available.
        Must not include any of [``shape``, ``chunks``, ``dtype``, ``store``].
    executor: str or rechunker.types.Executor
        Implementation of the execution engine for copying between zarr arrays.
        Supplying a custom Executor is currently even more experimental than the
        rest of Rechunker: we expect the interface to evolve as we add more
        executors and make no guarantees of backwards compatibility.

    Returns
    -------
    rechunked : :class:`Rechunked` object
    """
    if isinstance(executor, str):
        executor = _get_executor(executor)
    copy_spec, intermediate, target = _setup_rechunk(
        source=source,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        target_options=target_options,
        temp_store=temp_store,
        temp_options=temp_options,
    )
    plan = executor.prepare_plan(copy_spec)
    return Rechunked(executor, plan, source, intermediate, target)


def _setup_rechunk(
    source,
    target_chunks,
    max_mem,
    target_store,
    target_options=None,
    temp_store=None,
    temp_options=None,
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
                target_options=target_options,
                temp_store_or_group=temp_group,
                temp_options=temp_options,
                name=array_name,
            )
            copy_specs.append(copy_spec)

        return copy_specs, temp_group, target_group

    elif isinstance(source, (zarr.core.Array, dask.array.Array)):

        copy_spec = _setup_array_rechunk(
            source,
            target_chunks,
            max_mem,
            target_store,
            target_options=target_options,
            temp_store_or_group=temp_store,
            temp_options=temp_options,
        )
        intermediate = copy_spec.intermediate.array
        target = copy_spec.write.array
        return [copy_spec], intermediate, target

    else:
        raise ValueError("Source must be a Zarr Array or Group, or a Dask Array.")


def _validate_options(options):
    if not options:
        return
    for k in ["shape", "chunks", "dtype", "store", "name"]:
        if k in options:
            raise ValueError(
                f"Optional array arguments must not include {k} (provided {k}={options[k]}). "
                "Values for this property are managed internally."
            )


def _setup_array_rechunk(
    source_array,
    target_chunks,
    max_mem,
    target_store_or_group,
    target_options=None,
    temp_store_or_group=None,
    temp_options=None,
    name=None,
) -> CopySpec:
    _validate_options(target_options)
    _validate_options(temp_options)
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
        shape,
        target_store_or_group,
        target_chunks,
        dtype,
        name=name,
        **(target_options or {}),
    )
    try:
        target_array.attrs.update(source_array.attrs)
    except AttributeError:
        pass

    if read_chunks == write_chunks:
        int_array = None
    else:
        # do intermediate store
        assert temp_store_or_group is not None
        int_array = _zarr_empty(
            shape,
            temp_store_or_group,
            int_chunks,
            dtype,
            name=name,
            **(temp_options or {}),
        )

    read_proxy = ArrayProxy(source_array, read_chunks)
    int_proxy = ArrayProxy(int_array, int_chunks)
    write_proxy = ArrayProxy(target_array, write_chunks)
    return CopySpec(read_proxy, int_proxy, write_proxy)
