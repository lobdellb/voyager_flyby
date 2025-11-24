from __future__ import annotations

import pathlib
from typing import Dict, Iterable

import helpers
import pipeline


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
