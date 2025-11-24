import json
import pathlib

import pytest

import pipeline
from models.pipeline import StageArtifact


class DummyTask(pipeline.Task):
    def __init__(self, name="dummy", *, depends_on=None, items=None):
        super().__init__(name=name, depends_on=depends_on or [])
        self._items = items or []

    def produce_items(self, context):
        return list(self._items)

    def item_key(self, context, item):
        return str(item)


class CustomProcessTask(DummyTask):
    def process(self, context, work_item):
        return {"key": work_item.key, "data": work_item.data}

    def result_payload(self, context, work_item, result):
        return result


@pytest.fixture
def pipeline_with_session(in_memory_sessionmaker):
    return pipeline.Pipeline(in_memory_sessionmaker)


def test_stable_hash_is_deterministic():
    data = {"b": 1, "a": 2}
    first = pipeline.stable_hash(data)
    second = pipeline.stable_hash({"a": 2, "b": 1})
    assert first == second
    # ensure json dumps is stable
    assert len(first) == 64


def test_stage_state_store_records_and_retrieves(in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    store = pipeline.StageStateStore(session)
    run = store.start_run(note="note")
    stage_run = store.start_stage_run(run, "stage")
    artifact, changed = store.record_artifact("stage", "item", "hash1", {"value": 1})
    assert changed is True
    assert store.get_artifact("stage", "item") == artifact

    # Update with same payload should mark as clean and not changed
    artifact.payload["value"] = 1
    artifact2, changed2 = store.record_artifact("stage", "item", "hash1", {"value": 1})
    assert artifact2 is artifact
    assert changed2 is False

    store.finish_stage_run(stage_run, status="completed", processed_items=1, skipped_items=0)
    store.finish_run(run, status="completed")
    session.commit()
    assert stage_run.status == "completed"


def test_iter_work_items_skips_clean_artifacts(in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    task = DummyTask(items=[1, 2, 3])
    context = pipeline.StageContext(pipeline.PipelineContext(session), task)

    # record artifact for item 2 to simulate cached work
    context.store.record_artifact(task.name, "2", pipeline.stable_hash(2), {"value": 2})
    session.commit()

    keys = []
    for item in task.iter_work_items(context):
        keys.append(item.key)
    assert keys == ["1", "3"]
    assert context.skipped_items == 1


def test_pipeline_respects_dependencies(pipeline_with_session):
    t1 = CustomProcessTask(name="first", items=["a"])
    t2 = CustomProcessTask(name="second", depends_on=["first"], items=["b"])
    pipeline_with_session.register(t1)
    pipeline_with_session.register(t2)

    pipeline_with_session.run()

    with pipeline_with_session._session_factory() as session:
        artifacts = session.query(StageArtifact).all()
        names = sorted({a.stage_name for a in artifacts})
        assert names == ["first", "second"]


def test_pipeline_expands_selected_with_deps(pipeline_with_session):
    t1 = DummyTask(name="a")
    t2 = DummyTask(name="b", depends_on=["a"])
    pipeline_with_session.register(t1)
    pipeline_with_session.register(t2)

    order = pipeline_with_session._execution_order(["b"])
    assert order == ["a", "b"]


@pytest.mark.parametrize(
    "items, expected_hash",
    [
        ([{"x": 1}], pipeline.stable_hash({"x": 1})),
        ([], None),
    ],
)
def test_work_item_hash_consistency(items, expected_hash):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import database

    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    task = DummyTask(items=items)
    context = pipeline.StageContext(pipeline.PipelineContext(Session()), task)
    for work_item in task.iter_work_items(context):
        assert work_item.input_hash == expected_hash
