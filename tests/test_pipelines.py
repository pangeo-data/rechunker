"""
Test ParallelPiplines and related executors
"""

import pytest

from rechunker.executors.dask import DaskPipelineExecutor
from rechunker.executors.prefect import PrefectPipelineExecutor
from rechunker.executors.python import PythonPipelineExecutor
from rechunker.types import Stage


@pytest.fixture
def example_pipeline(tmpdir_factory):

    tmp = tmpdir_factory.mktemp("pipeline_data")

    def func0():
        tmp.join("func0.log").ensure(file=True)
        assert not tmp.join("func1_a.log").check(file=True)

    stage0 = Stage(func0)

    def func1(arg):
        tmp.join(f"func1_{arg}.log").ensure(file=True)

    stage1 = Stage(func1, ["a", "b", 3])

    # MultiStagePipeline
    pipeline = [stage0, stage1]
    # ParallelPipelines
    pipelines = [pipeline]
    return pipelines, tmp


@pytest.mark.parametrize(
    "Executor", [PythonPipelineExecutor, DaskPipelineExecutor, PrefectPipelineExecutor]
)
def test_pipeline(example_pipeline, Executor):
    pipeline, tmpdir = example_pipeline
    executor = Executor()
    plan = executor.pipelines_to_plan(pipeline)
    executor.execute_plan(plan)
    for fname in ["func0.log", "func1_a.log", "func1_b.log", "func1_3.log"]:
        assert tmpdir.join(fname).check(file=True)
