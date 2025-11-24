import json
import sys
import tarfile
import types
from pathlib import Path

import pytest

import pipeline
from models.image import VoyagerImage
from steps.list_tar_files import ListTarFiles
from steps.extract_tar_members import ExtractTarMembers
from steps.load_and_store_metadata import LoadAndStoreMetadata
from steps.load_vicar_to_pickle import LoadVicarImageToPickle
from steps.compute_circle_centers import ComputeAndStoreJupyterCenters


class PickleableVicar:
    def __init__(self, path):
        self.path = path
        self.array = [[1]]


class FakeArray:
    def __init__(self):
        self.data = [[1, 2], [3, 4]]

    def squeeze(self):
        return self.data


class PickleableLoaded:
    def __init__(self):
        self.array = FakeArray()


class FakeCircle(list):
    def tolist(self):
        return list(self)


def build_context(session, task):
    return pipeline.StageContext(pipeline.PipelineContext(session), task)


def create_tar_with_members(tmp_path: Path) -> Path:
    tar_path = tmp_path / "sample.tar.gz"
    inner_dir = tmp_path / "sample" / "DATA"
    inner_dir.mkdir(parents=True)
    (inner_dir / "C1234567_GEOMED.LBL").write_text("LABEL")
    (inner_dir / "C1234567_GEOMED.IMG").write_text("IMAGE")
    with tarfile.open(tar_path, "w:gz") as tar:
        for file in inner_dir.glob("*"):
            tar.add(file, arcname=file.relative_to(tmp_path))
    return tar_path


def test_list_tar_files(tmp_path, in_memory_sessionmaker):
    tar_path = create_tar_with_members(tmp_path)
    task = ListTarFiles(tmp_path)
    ctx = build_context(in_memory_sessionmaker(), task)
    items = list(task.produce_items(ctx))
    assert items[0]["tar_file_path"] == str(tar_path)
    assert task.item_key(ctx, items[0]) == "sample"


def test_extract_tar_members_records_members(tmp_path, in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    store = pipeline.StageStateStore(session)
    tar_path = create_tar_with_members(tmp_path)

    list_task = ListTarFiles(tmp_path)
    list_ctx = build_context(session, list_task)
    for item in list_task.produce_items(list_ctx):
        store.record_artifact("list_tar_files", item["stem"], list_task.item_input_hash(list_ctx, item), item)
    session.commit()

    task = ExtractTarMembers(tmp_path)
    ctx = build_context(session, task)
    items = list(task.produce_items(ctx))
    assert any(i["suffix"] == "LBL" for i in items)
    assert any(i["suffix"] == "IMG" for i in items)


def test_load_and_store_metadata_calls_upsert(monkeypatch, tmp_path, in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    store = pipeline.StageStateStore(session)
    payload = {"product_id": "C1_GEOMED.IMG", "suffix": "LBL", "local_file_path": str(tmp_path / "C1_GEOMED.IMG"), "upstream_hash": "h"}
    store.record_artifact("extract_tar_members", "C1_GEOMED.IMG", "hash", payload)
    session.commit()

    called = {}

    def fake_upsert(sess, product_id, fn):
        called["product_id"] = product_id
        called["fn"] = fn

    monkeypatch.setattr("steps.load_and_store_metadata.upsert_image_metadata", fake_upsert)

    task = LoadAndStoreMetadata()
    ctx = build_context(session, task)
    items = list(task.produce_items(ctx))
    assert items[0]["product_id"] == "C1_GEOMED.IMG"

    work_item = next(task.iter_work_items(ctx))
    task.process(ctx, work_item)
    assert called["product_id"] == "C1_GEOMED.IMG"


def test_load_vicar_to_pickle_creates_pickle(tmp_path, monkeypatch, in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    store = pipeline.StageStateStore(session)
    artifact_payload = {"product_id": "C1_GEOMED.IMG", "suffix": "IMG", "local_file_path": str(tmp_path / "file.IMG"), "upstream_hash": "h"}
    (tmp_path / "file.IMG").write_text("IMAGE")
    store.record_artifact("extract_tar_members", "C1_GEOMED.IMG", "hash", artifact_payload)
    session.add(VoyagerImage(PRODUCT_ID="C1_GEOMED.IMG", LOCAL_FILENAME=str(tmp_path / "file.IMG")))
    session.commit()

    import steps.load_vicar_to_pickle as lvp

    monkeypatch.setattr(lvp, "vicar", types.SimpleNamespace(VicarImage=PickleableVicar))

    task = LoadVicarImageToPickle(tmp_path)
    ctx = build_context(session, task)
    work_item = next(task.iter_work_items(ctx))
    task.process(ctx, work_item)
    assert Path(work_item.data["local_file_path"]).exists()
    assert Path(tmp_path / "pickled_images" / "C1_GEOMED.p").exists()


def test_compute_circle_centers_updates_record(monkeypatch, tmp_path, in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    store = pipeline.StageStateStore(session)

    pickle_path = tmp_path / "image.p"
    import pickle

    with open(pickle_path, "wb") as fp:
        pickle.dump(PickleableLoaded(), fp)

    store.record_artifact(
        "load_vicar_to_pickle",
        "C1_GEOMED.IMG",
        "hash",
        {"product_id": "C1_GEOMED.IMG", "pickle_path": str(pickle_path)},
    )
    image = VoyagerImage(PRODUCT_ID="C1_GEOMED.IMG", LOCAL_IMAGE_PICKLE_FN=str(pickle_path))
    session.add(image)
    session.commit()

    monkeypatch.setattr("analysis.scale_image", lambda arr: arr)
    monkeypatch.setattr("analysis.prep_impage_for_cv2", lambda arr: arr)
    monkeypatch.setattr("analysis.find_circle_center_parametrized", lambda *a, **k: [[FakeCircle([10, 20, 5])]])

    task = ComputeAndStoreJupyterCenters(commit_every=1)
    ctx = build_context(session, task)
    work_item = next(task.iter_work_items(ctx))
    result = task.process(ctx, work_item)
    assert result["center"] == (10, 20)
    assert image.BEST_CIRCLE_X == 10
    assert image.BEST_CIRCLE_Y == 20
