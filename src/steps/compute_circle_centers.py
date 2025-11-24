from __future__ import annotations

import json
import time
from typing import Dict

import cv2

import analysis
import pipeline
from repository.image import get_voyager_image_by_product_id


class ComputeAndStoreJupyterCenters(pipeline.Task):
    """Compute circle centers for the pickled images."""

    def __init__(self, commit_every: int = 50):
        super().__init__(name="compute_circle_centers", depends_on=("load_vicar_to_pickle",))
        self.commit_every = commit_every
        self._since_commit = 0

    def produce_items(self, context: pipeline.StageContext):
        for artifact in context.store.artifacts_for("load_vicar_to_pickle"):
            product_id = artifact.payload["product_id"]
            image_record = get_voyager_image_by_product_id(context.session, product_id)
            if not image_record or image_record.LOCAL_IMAGE_PICKLE_FN is None:
                continue
            yield {
                "product_id": product_id,
                "pickle_path": image_record.LOCAL_IMAGE_PICKLE_FN,
                "upstream_hash": artifact.input_hash,
                "existing_center": (
                    image_record.BEST_CIRCLE_X,
                    image_record.BEST_CIRCLE_Y,
                ),
            }

    def item_key(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return item["product_id"]

    def item_input_hash(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return pipeline.stable_hash({"pickle_hash": item["upstream_hash"]})

    def process(self, context: pipeline.StageContext, work_item: pipeline.WorkItem):
        product_id = work_item.data["product_id"]
        image_record = get_voyager_image_by_product_id(context.session, product_id)
        if not image_record or not image_record.LOCAL_IMAGE_PICKLE_FN:
            raise ValueError(f"Pickled image missing for {product_id}")

        import pickle

        with open(image_record.LOCAL_IMAGE_PICKLE_FN, "rb") as fp:
            v_im = pickle.load(fp)

        im = analysis.scale_image(v_im.array.squeeze())
        im_prepped_for_cv2 = analysis.prep_impage_for_cv2(im)

        start_time = time.time()
        circles = analysis.find_circle_center_parametrized(
            im_prepped_for_cv2,
            blur_width=5,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=44,
            minRadius=25,
            maxRadius=0,
        )
        elapsed = time.time() - start_time

        if circles is None:
            return {
                "product_id": product_id,
                "status": "no_circles",
                "elapsed": elapsed,
            }

        best_circle = circles[0][0]
        image_record.BEST_CIRCLE_X = int(best_circle[0])
        image_record.BEST_CIRCLE_Y = int(best_circle[1])
        image_record.ALL_CIRCLES = json.dumps([c.tolist() for c in circles[0]])
        image_record.CIRCLE_TIME = float(elapsed)

        self._since_commit += 1
        if self._since_commit >= self.commit_every:
            context.session.commit()
            self._since_commit = 0

        return {
            "product_id": product_id,
            "center": (int(best_circle[0]), int(best_circle[1])),
            "elapsed": elapsed,
        }

    def on_stage_completed(self, context: pipeline.StageContext) -> None:
        if self._since_commit:
            context.session.commit()
            self._since_commit = 0
