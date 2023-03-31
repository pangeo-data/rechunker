"""
Test ParallelPiplines and related executors
"""

import pytest

# TODO: remove the hard dependency on prefect here
pytest.importorskip("prefect")

from dataclasses import dataclass

from rechunker.executors.dask import DaskPipelineExecutor
from rechunker.executors.prefect import PrefectPipelineExecutor
from rechunker.executors.python import PythonPipelineExecutor
from rechunker.types import Pipeline, Stage


@pytest.fixture
def example_pipeline(tmpdir_factory):
    tmp = tmpdir_factory.mktemp("pipeline_data")

    @dataclass(frozen=True)
    class Config:
        fname0: str
        fname1: str
        fname_pattern: str

    config = Config("func0.log", "func1_a.log", "func1_{arg}.log")

    def func0(config=Config):
        tmp.join(config.fname0).ensure(file=True)
        assert not tmp.join(config.fname1).check(file=True)

    stage0 = Stage(func0, "write_some_files")

    def func1(arg, config=Config):
        fname = config.fname_pattern.format(arg=arg)
        tmp.join(fname).ensure(file=True)

    stage1 = Stage(func1, "write_many_files", mappable=["a", "b", 3])
    pipeline = Pipeline(stages=[stage0, stage1], config=config)
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
