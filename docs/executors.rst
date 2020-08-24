.. _executors:

Executors
=========

``rechunker`` plans can be executed by a variety of executors. The default is ``dask``, which executes the plan on a Dask cluster.

For example, we can use :ref:`rechunker.executors.python.PythonExecutor` to execute a plan as a simple Python for loop, which might be useful for debugging.

.. code-block:: python

  >>> import zarr
  >>> from rechunker import rechunk
  >>> source = zarr.ones((4, 4), chunks=(2, 2), store="source.zarr")
  >>> intermediate = "intermediate.zarr"
  >>> target = "target.zarr"
  >>> rechunked = rechunk(
  ...    source, target_chunks=(4, 1), target_store=target,
  ...    max_mem=256000000, temp_store=intermediate,
  ...    executor="python"
  ... )
  >>> rechunked.execute()

.. note::

   Most executors will require installing additional optional
   dependencies.


See :ref:`api.executors` for a list of all the different executors.
