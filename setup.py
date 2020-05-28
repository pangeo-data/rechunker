import os
import versioneer
from setuptools import find_packages, setup

here = os.path.dirname(__file__)

install_requires = []
extras_require = {
    "complete": ["dask[array]", "zarr", "pyyaml", "fsspec"],
}
extras_require["dev"] = extras_require["complete"] + [
    "pytest",
    "hypothesis",
    "flake8",
    "black",
]

setup(
    name="rechunker",
    description="A library for rechunking arrays.",
    long_description="A library for rechunking arrays.",
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
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
