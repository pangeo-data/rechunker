Release Notes
=============

v0.5.3 - 2025-03-20
-------------------

- Fix for dask>=2024.12.0. By `Pascal Bourgault <https://github.com/aulemahal>`_.
- Fix for xarray>=2024.10.0 with datasets opened directly from disk. By `Pascal Bourgault <https://github.com/aulemahal>`_.

v0.5.2 - 2023-08-12
-------------------

- Modernize the packaging of rechunker a bit. By `Filipe Fernandes <https://github.com/ocefpaf>`_.
- Allow rechunking to groups. By `Ryan Abernathey <https://github.com/rabernat>`_.

v0.5.1 - 2023-03-14
-------------------

- Properly copy all attributes. By `Matt Savoie <https://github.com/flamingbear>`_.
- Fix regression with Zarr + fsspec. By `Ryan Abernathey <https://github.com/rabernat>`_.


v0.5.0 - 2022-04-14
-------------------

- Fix major bug with dask executor.
  By `Ryan Abernathey <https://github.com/rabernat>`_.
- Enable xarray ``.chunk()`` style input for target chunks.
  By `Julius Busecke <https://github.com/jbusecke>`_.

v0.4.2 - 2021-04-27
-------------------

- Fix dependency bug with v0.4 correctly.

v0.4.1 - 2021-04-26
-------------------

- Fix dependency bug with v0.4.

v0.4 - 2021-04-19
-----------------

- Internal refactor of Executor class that allows for reuse in outside projects
  (e.g. Pangeo Forge). By `Ryan Abernathey <https://github.com/rabernat>`_.


v0.3.3 - 2021-01-15
-------------------

- Fixed bug preventing specification of ``target_chunks`` as dict with Xarray inputs.
  By `Ryan Abernathey <https://github.com/rabernat>`_
- Fixed bug in interaction with fsspec stores.
  By `Landung Setiawan <https://github.com/lsetiawan>`_


v0.3.2 - 2020-12-02
-------------------

- Fixed bug in rechunking of xarray datasets. By `Filipe Fernandes <https://github.com/ocefpaf>`_.
- Internal improvements to tests and packagaging. By `Filipe Fernandes <https://github.com/ocefpaf>`_.
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
