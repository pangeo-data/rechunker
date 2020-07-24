from typing import Any, Iterable, NamedTuple, Tuple

import zarr


class CopySpec(NamedTuple):
    """Specifcation of how to copy between two arrays."""
    # TODO: remove Any by making CopySpec a Generic, once we only support Python
    # 3.7+: https://stackoverflow.com/questions/50530959
    source: Any
    target: zarr.Array
    chunks: Tuple[int, ...]


class StagedCopySpec:
    stages: Tuple[CopySpec, ...]

    def __init__(self, stages: Iterable[CopySpec]):
        self.stages = tuple(stages)
