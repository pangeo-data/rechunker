from typing import Iterable, NamedTuple, Tuple, Union

import dask.array
import zarr


class CopySpec(NamedTuple):
    source: Union[zarr.Array, dask.array.Array]
    target: zarr.Array
    chunks: Tuple[int, ...]


class StagedCopySpec:
    stages: Tuple[CopySpec, ...]

    def __init__(self, stages: Iterable[CopySpec]):
        self.stages = tuple(stages)
