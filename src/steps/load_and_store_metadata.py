from __future__ import annotations

from typing import Dict

import pipeline
from repository.image import upsert_image_metadata


class LoadAndStoreMetadata(pipeline.Task):
    """Load PVL label files into the Voyager images table."""

    def __init__(self):
        super().__init__(name="load_and_store_metadata", depends_on=("extract_tar_members",))

    def produce_items(self, context: pipeline.StageContext):
        for artifact in context.store.artifacts_for("extract_tar_members"):
            payload = artifact.payload
            if payload.get("suffix") != "LBL":
                continue
            enriched = dict(payload)
            enriched["upstream_hash"] = artifact.input_hash
            yield enriched

    def item_key(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return item["product_id"]

    def item_input_hash(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return pipeline.stable_hash({"label_hash": item["upstream_hash"]})

    def process(self, context: pipeline.StageContext, work_item: pipeline.WorkItem):
        local_path = work_item.data["local_file_path"].replace(".IMG", ".LBL")
        product_id = work_item.data["product_id"]
        upsert_image_metadata(context.session, product_id=product_id, fn=local_path)
        return {"product_id": product_id, "label_path": local_path}
