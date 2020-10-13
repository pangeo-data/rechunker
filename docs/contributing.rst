Contributing
============

Releasing
---------

On the master branch, with a clean checkout

.. code-block::

   git commit --allow-empty -m "RLS: v0.2.0"
   git tag -a 0.2.0 -m "RLS: 0.2.0"
   git push upstream master --follow-tags

A GitHub Action should push the release to PyPI. After that, conda-forge's
bot will update the conda packages.
