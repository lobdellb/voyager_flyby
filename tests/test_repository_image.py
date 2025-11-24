import json
import types

import pytest

import repository.image as repo
from models.image import VoyagerImage


def test_handle_special_cases_converts_unknown():
    data = {"START_TIME": "UNK", "OTHER": 1}
    result = repo.handle_special_cases(data)
    assert result["START_TIME"] is None
    assert result["OTHER"] == 1


def test_flatten_vicar_object_handles_nested_and_quantity():
    # Build fake PVL structure
    quantity = repo.pvl.collections.Quantity(10, "km")
    inner = repo.pvl.collections.PVLObject([("VALUE", quantity)])
    module = repo.pvl.collections.PVLModule(
        [
            ("TOP", "val"),
            ("INNER", inner),
            ("LIST", []),
        ]
    )
    result = repo.flatten_vicar_object(module, to_exclude=["LIST"])
    assert result == {
        "TOP": "val",
        "INNER_VALUE_value": 10,
        "INNER_VALUE_units": "km",
    }


def test_voyager_image_from_dict_filters_unknown_keys():
    payload = {"PRODUCT_ID": "PID", "EXTRA": "ignored"}
    image = repo.voyager_image_from_dict(payload)
    assert isinstance(image, VoyagerImage)
    assert image.PRODUCT_ID == "PID"
    assert not hasattr(image, "EXTRA")


def test_get_voyager_image_by_product_id(in_memory_sessionmaker):
    session = in_memory_sessionmaker()
    image = VoyagerImage(PRODUCT_ID="ABC")
    session.add(image)
    session.commit()

    found = repo.get_voyager_image_by_product_id(session, "ABC")
    missing = repo.get_voyager_image_by_product_id(session, "MISSING")
    assert found.PRODUCT_ID == "ABC"
    assert missing is False


def test_upsert_image_metadata_reads_file(tmp_path, monkeypatch, in_memory_sessionmaker):
    session = in_memory_sessionmaker()

    # Write minimal PVL-like content
    label_path = tmp_path / "example.LBL"
    label_path.write_text("PRODUCT_ID = SAMPLE\nDATA_SET_ID = SET")

    called = {}

    def fake_loads(text):
        called["text"] = text
        module = repo.pvl.collections.PVLModule([("PRODUCT_ID", "SAMPLE"), ("DATA_SET_ID", "SET")])
        return module

    monkeypatch.setattr(repo.pvl, "loads", fake_loads)

    image = repo.upsert_image_metadata(session, "SAMPLE_GEOMED.IMG", str(label_path))
    assert image.PRODUCT_ID == "SAMPLE"
    assert called["text"] == label_path.read_text()
    assert repo.get_voyager_image_by_product_id(session, "SAMPLE")
