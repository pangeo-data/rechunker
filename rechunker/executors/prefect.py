from typing import Iterable, Tuple

import prefect

from rechunker.executors.util import chunk_keys
from rechunker.types import (
    StagedCopySpec,
    Executor,
    ReadableArray,
    WriteableArray,
)


class PrefectExecutor(Executor[prefect.Flow]):
    """An execution engine based on Prefect.

    Supports copying between any arrays that implement ``__getitem__`` and
    ``__setitem__`` for tuples of ``slice`` objects. Array must also be
    serializable by Beam (i.e., with pickle).

    Execution plans for PrefectExecutor are prefect.Flow objects.
    """

    def prepare_plan(self, specs: Iterable[StagedCopySpec]) -> prefect.Flow:
        return _make_flow(specs)

    def execute_plan(self, plan: prefect.Flow, **kwargs):
        return plan.run(**kwargs)


@prefect.task
def _copy_chunk(
    source: ReadableArray, target: WriteableArray, key: Tuple[int, ...]
) -> None:
    target[key] = source[key]


def _make_flow(specs: Iterable[StagedCopySpec]) -> prefect.Flow:
    with prefect.Flow("Rechunker") as flow:
        # iterate over different arrays in the group
        for staged_copy_spec in specs:
            copy_tasks = []
            for source, target, chunks in staged_copy_spec.stages:
                keys = list(chunk_keys(source.shape, chunks))
                copy_task = _copy_chunk.map(
                    prefect.unmapped(source), prefect.unmapped(target), keys
                )
                copy_tasks.append(copy_task)
            # create dependence between stages
            for n in range(len(copy_tasks) - 1):
                copy_tasks[n + 1].set_upstream(copy_tasks[n])
    return flow
