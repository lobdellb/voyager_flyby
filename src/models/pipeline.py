import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import relationship

import database


class PipelineRun(database.Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    finished_at = Column(DateTime(timezone=True))
    status = Column(String, default="running", nullable=False)
    note = Column(String)

    stage_runs = relationship("StageRun", back_populates="run", cascade="all, delete-orphan")


class StageRun(database.Base):
    __tablename__ = "pipeline_stage_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    stage_name = Column(String, nullable=False, index=True)
    status = Column(String, default="running", nullable=False)
    started_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    finished_at = Column(DateTime(timezone=True))
    processed_items = Column(Integer, default=0)
    skipped_items = Column(Integer, default=0)
    message = Column(String)

    run = relationship("PipelineRun", back_populates="stage_runs")


class StageArtifact(database.Base):
    __tablename__ = "pipeline_stage_artifacts"
    __table_args__ = (
        UniqueConstraint("stage_name", "item_key", name="uq_pipeline_stage_item"),
    )

    id = Column(Integer, primary_key=True)
    stage_name = Column(String, nullable=False, index=True)
    item_key = Column(String, nullable=False)
    input_hash = Column(String, nullable=False)
    payload = Column(JSON)
    status = Column(String, default="complete", nullable=False)
    needs_rerun = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    def mark_updated(self):
        self.updated_at = datetime.datetime.utcnow()
        self.status = "complete"
        self.needs_rerun = False
