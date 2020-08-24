API
===

Rechunk Function
----------------

The main function exposed by rechunker is :func:`rechunker.rechunk`.

.. currentmodule:: rechunker

.. autofunction:: rechunk


The Rechunked Object
--------------------

``rechunk`` returns a :class:`Rechunked` object.

.. autoclass:: Rechunked

.. note::
   You must call ``execute()`` on the ``Rechunked`` object in order to actually
   perform the rechunking operation.

.. warning::
   You must manually delete the intermediate store when ``execute`` is finished.


.. _api.executors:

Executors
---------

Rechunking plans can be executed on a variety of backends. The following table lists the current options.

.. autosummary::

   rechunker.executors.beam.BeamExecutor
   rechunker.executors.dask.DaskExecutor
   rechunker.executors.prefect.PrefectExecutor
   rechunker.executors.python.PythonExecutor
   rechunker.executors.pywren.PywrenExecutor

.. autoclass:: rechunker.executors.beam.BeamExecutor
.. autoclass:: rechunker.executors.dask.DaskExecutor
.. autoclass:: rechunker.executors.prefect.PrefectExecutor
.. autoclass:: rechunker.executors.python.PythonExecutor
.. autoclass:: rechunker.executors.pywren.PywrenExecutor
