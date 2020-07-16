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
