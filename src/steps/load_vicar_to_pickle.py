from __future__ import annotations

import pathlib
from typing import Dict

import vicar

import helpers
import pipeline
from repository.image import get_voyager_image_by_product_id


class LoadVicarImageToPickle(pipeline.Task):
    """Persist VICAR images to local pickle files for faster downstream access."""

    def __init__(self, cache_path: pathlib.Path, commit_every: int = 50):
        super().__init__(
            name="load_vicar_to_pickle",
            depends_on=("extract_tar_members", "load_and_store_metadata"),
        )
        self.cache_path = pathlib.Path(cache_path)
        self.commit_every = commit_every
        self._since_commit = 0
        (self.cache_path / "pickled_images").mkdir(parents=True, exist_ok=True)

    def produce_items(self, context: pipeline.StageContext):
        for artifact in context.store.artifacts_for("extract_tar_members"):
            payload = artifact.payload
            if payload.get("suffix") != "IMG":
                continue
            enriched = dict(payload)
            enriched["upstream_hash"] = artifact.input_hash
            yield enriched

    def item_key(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return item["product_id"]

    def item_input_hash(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return pipeline.stable_hash({"image_hash": item["upstream_hash"]})

    def process(self, context: pipeline.StageContext, work_item: pipeline.WorkItem):
        product_id = work_item.data["product_id"]
        image_record = get_voyager_image_by_product_id(context.session, product_id)
        if not image_record:
            raise ValueError(f"Image metadata for {product_id} must exist before loading pickles")

        image_id = helpers.extract_prefix_from_filename(product_id)
        pickle_path = self.cache_path / "pickled_images" / f"{image_id}_GEOMED.p"

        needs_generation = not pickle_path.exists()
        if work_item.artifact and work_item.artifact.input_hash != work_item.input_hash:
            needs_generation = True
        if pickle_path.exists() and pickle_path.stat().st_size == 0:
            needs_generation = True

        if needs_generation:
            v_im = vicar.VicarImage(image_record.LOCAL_FILENAME)
            with open(pickle_path, "wb") as fp:
                import pickle

                pickle.dump(v_im, fp)

            image_record.LOCAL_IMAGE_PICKLE_FN = str(pickle_path)
            self._since_commit += 1
            if self._since_commit >= self.commit_every:
                context.session.commit()
                self._since_commit = 0
        elif image_record.LOCAL_IMAGE_PICKLE_FN != str(pickle_path):
            image_record.LOCAL_IMAGE_PICKLE_FN = str(pickle_path)
            self._since_commit += 1
            if self._since_commit >= self.commit_every:
                context.session.commit()
                self._since_commit = 0

        return {
            "product_id": product_id,
            "pickle_path": str(pickle_path),
        }

    def on_stage_completed(self, context: pipeline.StageContext) -> None:
        if self._since_commit:
            context.session.commit()
            self._since_commit = 0
