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
scheme, to a new location.

Rechunker is designed to be used within a parallel execution framework such as
Dask_.

Usage
-----

The main function exposed by rechunker is :func:`rechunker.rechunk`.

.. currentmodule:: rechunker

.. autofunction:: rechunk

``rechunk`` returns a :class:`Rechunked` object.

.. autoclass:: Rechunked


Examples
--------


.. warning::
   You must manually delete the intermediate store when rechunker is finished
   executing.


The Rechunker Algorithm
-----------------------

The algorithm used by rechunker tries to satisfy several constraints simultaneously:

- *Respect memory limits.* Rechunker's algorithm guarantees that worker processes
  will not exceed a user-specified memory threshold.
- *Minimize the number of required tasks.* Specificallly, for N source chunks
  and M target chunks, the number of tasks is always less than N + M.
- *Be embarassingly parallel.* The task graph should be as simple as possible,
  to make it easy to execute using different task scheduling frameworks. This also
  means avoiding write locks, which are complex to manage.


.. _Zarr: https://zarr.readthedocs.io/en/stable/
.. _TileDB: https://tiledb.com/
.. _Dask: https://dask.org/
