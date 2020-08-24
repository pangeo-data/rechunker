.. Rechunker documentation master file, created by
   sphinx-quickstart on Mon Jul 13 17:10:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rechunker
=========

Rechunker is a Python package which enables efficient and scalable manipulation
of the chunk structure of chunked array formats such as Zarr_ and TileDB_.
Rechunker takes an input array (or group of arrays) stored in a persistent
storage device (such as a filesystem or a cloud storage bucket) and writes
out an array (or group of arrays) with the same data, but different chunking
scheme, to a new location. Rechunker is designed to be used within a parallel
execution framework such as Dask_.


Quickstart
----------

To install::

  >>> pip install rechunker


To use::

  >>> import zarr
  >>> from rechunker import rechunk
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


Contents
--------

.. toctree::
   :maxdepth: 2

   tutorial
   api
   algorithm
   executors
   release_notes
   contributing


.. _Zarr: https://zarr.readthedocs.io/en/stable/
.. _TileDB: https://tiledb.com/
.. _Dask: https://dask.org/
