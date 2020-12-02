Release Notes
=============

v0.3.2 - 2020-12-02
-------------------

- Fixed bug in rechunking of xarray datasets. By `Filipe Fernandes <https://github.com/ocefpaf>`_.
- Internal improvments to tests and packagaging. By `Filipe Fernandes <https://github.com/ocefpaf>`_.
- Updates to tutorial. By `Andrew Brettin <https://github.com/andrewbrettin>`_.

v0.3.1 - 2020-10-13
-------------------

Note: Skipped 0.3.0 due to error in release workflow.

- Added options for Zarr array definition. By `Eric Czech <https://github.com/eric-czech>`_.
- Exerimental support for rechunking Xarray datasets. By `Eric Czech <https://github.com/eric-czech>`_.
- Better internal type checking. By `Tom White <https://github.com/tomwhite>`_.

v0.2.0 - 2020-08-24
-------------------

- Added ``rechunker.executors`` for executing plans with other
  backends like Apache Beam, prefect, and pywren. See :ref:`executors` for more.
- Fixed overflow bug when computing the number of chunks needed for a memory target.
- Documentation update and tutorial.
- Allow rechunk to accept a Dask array.


v0.0.1 - 2020-07-15
-------------------

First Release
