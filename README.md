Rechunker
=========

[![Documentation Status](https://readthedocs.org/projects/rechunker/badge/?version=latest)](https://rechunker.readthedocs.io/en/latest/?badge=latest)
![Tests](https://github.com/pangeo-data/rechunker/workflows/Tests/badge.svg)
[![Coverage](https://codecov.io/github/pangeo-data/rechunker/coverage.svg?branch=master)](https://codecov.io/github/pangeo-data/rechunker?branch=master)


`Rechunker` is a Python package which enables efficient and scalable manipulation of the chunk structure of chunked array formats such as [Zarr](https://zarr.readthedocs.io/en/stable/) and [TileDB](https://tiledb.com/). `Rechunker` takes an input array (or group of arrays) stored in a persistent storage device (such as a filesystem or a cloud storage bucket) and writes out an array (or group of arrays) with the same data, but different chunking scheme, to a new location.

Rechunker is designed to be used within a parallel execution framework such as [Dask](https://dask.org/).

See [the documentation](https://rechunker.readthedocs.io/en/latest/) for more.
