from __future__ import annotations

import json
import logging
import os
import pathlib
import tarfile
import time
from typing import Dict, Iterable

import cv2
import vicar

import analysis
import config
import helpers
import pipeline
import database as db
from repository.image import get_voyager_image_by_product_id, upsert_image_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db.Base.metadata.create_all(bind=db.engine)


class ListTarFiles(pipeline.Task):
    """Enumerate raw Voyager tar archives."""

    def __init__(self, source_path: pathlib.Path):
        super().__init__(name="list_tar_files")
        self.source_path = pathlib.Path(source_path)

    def produce_items(self, context: pipeline.StageContext) -> Iterable[Dict[str, str]]:
        for tar_path in sorted(self.source_path.glob("*.tar.gz")):
            stats = tar_path.stat()
            yield {
                "tar_file_path": str(tar_path),
                "stem": helpers.extract_stem(tar_path),
                "size": stats.st_size,
                "mtime": stats.st_mtime,
            }

    def item_key(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return item["stem"]

    def item_input_hash(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return pipeline.stable_hash({
            "path": item["tar_file_path"],
            "size": item["size"],
            "mtime": item["mtime"],
        })


class ExtractTarMembers(pipeline.Task):
    """Extract the individual VICAR label and image files from tarballs."""

    def __init__(self, cache_path: pathlib.Path):
        super().__init__(name="extract_tar_members", depends_on=("list_tar_files",))
        self.cache_path = pathlib.Path(cache_path)
        self.members_root = self.cache_path / "tar_members"
        self.members_root.mkdir(parents=True, exist_ok=True)

    def produce_items(self, context: pipeline.StageContext):
        for tar_artifact in context.store.artifacts_for("list_tar_files"):
            tar_path = pathlib.Path(tar_artifact.payload["tar_file_path"])
            stem = tar_artifact.payload["stem"]
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    description = self._describe_member(tar_path, stem, member)
                    if not description:
                        continue
                    description["upstream_hash"] = tar_artifact.input_hash
                    yield description

    def item_key(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return item["member_name"]

    def item_input_hash(self, context: pipeline.StageContext, item: Dict[str, str]) -> str:
        return pipeline.stable_hash({
            "tar": item["tar_file_path"],
            "member": item["member_name"],
            "size": item["member_size"],
            "mtime": item["member_mtime"],
        })

    def process(self, context: pipeline.StageContext, work_item: pipeline.WorkItem):
        data = work_item.data
        local_path = pathlib.Path(data["local_file_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)

        should_extract = not local_path.exists()
        if work_item.artifact and work_item.artifact.input_hash != work_item.input_hash:
            should_extract = True
        if local_path.exists() and local_path.stat().st_size != data["member_size"]:
            should_extract = True

        if should_extract:
            with tarfile.open(data["tar_file_path"], "r") as tar:
                member = tar.getmember(data["member_name"])
                tar.extract(member, path=self.members_root)

        data["local_file_path"] = str(local_path)
        return data

    def result_payload(self, context: pipeline.StageContext, work_item: pipeline.WorkItem, result):
        return result

    def _describe_member(self, tar_path: pathlib.Path, stem: str, member: tarfile.TarInfo):
        member_name = member.name
        inner_path = pathlib.Path(member_name)
        suffix = inner_path.suffix.replace(".", "")
        inner_stem = inner_path.stem
        if "_" in inner_stem:
            image_id, image_type = inner_stem.split("_", 1)
        else:
            image_id, image_type = None, None

        if not self._should_keep(member_name, stem, suffix, image_type):
            return None

        local_file_path = self.members_root / member_name
        product_id = f"{image_id}_GEOMED.IMG" if image_id else inner_path.name
        return {
            "tar_file_path": str(tar_path),
            "stem": stem,
            "member_name": member_name,
            "member_size": member.size,
            "member_mtime": member.mtime,
            "suffix": suffix,
            "image_id": image_id,
            "image_type": image_type,
            "product_id": product_id,
            "local_file_path": str(local_file_path),
        }

    def _should_keep(self, member_name: str, stem: str, suffix: str, image_type: str | None) -> bool:
        return (
            member_name.startswith(f"{stem}/DATA/")
            and suffix in {"IMG", "LBL"}
            and image_type == "GEOMED"
        )


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


def build_pipeline() -> pipeline.Pipeline:
    runner = pipeline.Pipeline(db.SessionLocal)
    runner.register(ListTarFiles(config.source_path))
    runner.register(ExtractTarMembers(config.cache_path))
    runner.register(LoadAndStoreMetadata())
    runner.register(LoadVicarImageToPickle(config.cache_path))
    runner.register(ComputeAndStoreJupyterCenters())
    return runner


def main(selected_stages: Iterable[str] | None = None):
    runner = build_pipeline()
    runner.run(selected_stages=selected_stages)


if __name__ == "__main__":
    logger.info("Python path is: %%s", os.getenv("PYTHONPATH"))
    logger.info("System path is %s", os.getcwd())
    main()
    logger.info("Done")
