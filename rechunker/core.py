from typing import Iterable, NamedTuple, Tuple

import zarr


class CopySpec(NamedTuple):
    source: zarr.Array
    target: zarr.Array
    chunks: Tuple[int, ...]


class StagedCopySpec:
    stages: Tuple[CopySpec, ...]

    def __init__(self, stages: Iterable[CopySpec]):
        self.stages = tuple(stages)
