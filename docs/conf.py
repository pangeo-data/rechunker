# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import sys

print("python exec:", sys.executable)
print("sys.path:", sys.path)


import rechunker
import sphinx_pangeo_theme  # noqa: F401


# -- Project information -----------------------------------------------------

project = "Rechunker"
copyright = "2020, Ryan Abernathey, Tom Augspurger"
author = "Ryan Abernathey, Tom Augspurger"

# The full version, including alpha/beta/rc tags
release = rechunker.__version__
# The short X.Y version.
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "nbsphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxcontrib.srclinks",
]

# https://nbsphinx.readthedocs.io/en/0.2.14/never-execute.html
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

intersphinx_mapping = {
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pangeo"

html_sidebars = {
    "index": ["localtoc.html", "srclinks.html"],
    "**": ["localtoc.html", "srclinks.html"],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sourcelink = True
srclink_project = "https://github.com/pangeo-data/rechunker"
srclink_branch = "master"
srclink_src_path = "docs/"
