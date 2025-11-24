from __future__ import annotations

import logging
import os
from typing import Iterable

import config
import database as db
import pipeline
from steps import (
    ComputeAndStoreJupyterCenters,
    ExtractTarMembers,
    ListTarFiles,
    LoadAndStoreMetadata,
    LoadVicarImageToPickle,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db.Base.metadata.create_all(bind=db.engine)


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
    logger.info("Python path is: %s", os.getenv("PYTHONPATH"))
    logger.info("System path is %s", os.getcwd())
    main()
    logger.info("Done")
