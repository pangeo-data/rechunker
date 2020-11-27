import os
from setuptools import find_packages, setup
from setuptools import setup

here = os.path.dirname(__file__)
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


install_requires = ["dask[array]", "zarr", "xarray"]
doc_requires = [
    "sphinx",
    "sphinxcontrib-srclinks",
    "sphinx-pangeo-theme",
    "numpydoc",
    "IPython",
    "nbsphinx",
]

extras_require = {
    "complete": install_requires
    + ["apache_beam", "pyyaml", "fsspec", "prefect", "pywren_ibm_cloud"],
    "docs": doc_requires,
}
extras_require["dev"] = extras_require["complete"] + [
    "pytest",
    "pytest-cov",
    "hypothesis",
    "flake8",
    "black",
    "codecov",
    "mypy==0.782",
]

setup(
    name="rechunker",
    description="A library for rechunking arrays.",
    url="https://github.com/pangeo-data/rechunker",
    author="Pangeo developers",
    author_email="ryan.abernathey@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires="setuptools_scm",
    use_scm_version={
        "write_to": "rechunker/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
