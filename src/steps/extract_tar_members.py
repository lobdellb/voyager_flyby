from __future__ import annotations

import pathlib
import tarfile
from typing import Dict, Iterable

import helpers
import pipeline


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
