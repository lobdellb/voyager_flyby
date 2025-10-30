"""Pipeline orchestration primitives.

This module provides a small framework for defining enrichment stages that
coordinate through the database.  Each task expresses its dependencies and the
pipeline runner records run history, item-level artifacts, and dependency
invalidations so that work is only performed when inputs change.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

from sqlalchemy.orm import Session

from models.pipeline import PipelineRun, StageArtifact, StageRun

logger = logging.getLogger(__name__)


def _stable_json_dumps(data: Any) -> str:
    """Serialize *data* into a deterministic JSON string."""

    return json.dumps(data, sort_keys=True, default=str)


def stable_hash(data: Any) -> str:
    """Return a stable hash for arbitrary JSON-serialisable data."""

    payload = _stable_json_dumps(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class WorkItem:
    """A single unit of work for a stage."""

    key: str
    data: Any
    input_hash: str
    artifact: Optional[StageArtifact] = None


class StageStateStore:
    """Persists run metadata and item-level artifacts for the pipeline."""

    def __init__(self, session: Session):
        self.session = session

    # -- Run bookkeeping -------------------------------------------------
    def start_run(self, note: Optional[str] = None) -> PipelineRun:
        run = PipelineRun(note=note)
        self.session.add(run)
        self.session.flush()
        return run

    def finish_run(self, run: PipelineRun, status: str) -> None:
        run.status = status
        run.finished_at = datetime.datetime.utcnow()

    def fail_run(self, run: PipelineRun, message: Optional[str] = None) -> None:
        run.status = "failed"
        run.finished_at = datetime.datetime.utcnow()
        if message:
            run.note = (run.note + "\n" + message) if run.note else message

    # -- Stage bookkeeping -----------------------------------------------
    def start_stage_run(self, run: PipelineRun, stage_name: str) -> StageRun:
        stage_run = StageRun(run=run, stage_name=stage_name)
        self.session.add(stage_run)
        self.session.flush()
        return stage_run

    def finish_stage_run(
        self,
        stage_run: StageRun,
        *,
        status: str,
        processed_items: int,
        skipped_items: int,
        message: Optional[str] = None,
    ) -> None:
        stage_run.status = status
        stage_run.finished_at = datetime.datetime.utcnow()
        stage_run.processed_items = processed_items
        stage_run.skipped_items = skipped_items
        stage_run.message = message

    def fail_stage_run(self, stage_run: StageRun, message: Optional[str] = None) -> None:
        stage_run.status = "failed"
        stage_run.finished_at = datetime.datetime.utcnow()
        stage_run.message = message

    # -- Artifact helpers ------------------------------------------------
    def get_artifact(self, stage_name: str, item_key: str) -> Optional[StageArtifact]:
        return (
            self.session.query(StageArtifact)
            .filter(StageArtifact.stage_name == stage_name, StageArtifact.item_key == item_key)
            .one_or_none()
        )

    def artifacts_for(self, stage_name: str) -> List[StageArtifact]:
        return (
            self.session.query(StageArtifact)
            .filter(StageArtifact.stage_name == stage_name)
            .order_by(StageArtifact.item_key)
            .all()
        )

    def mark_artifact_clean(self, artifact: StageArtifact) -> None:
        artifact.mark_updated()

    def record_artifact(
        self,
        stage_name: str,
        item_key: str,
        input_hash: str,
        payload: Any,
    ) -> tuple[StageArtifact, bool]:
        """Insert or update an artifact record.

        Returns a tuple of the ORM object and a boolean indicating whether the
        payload changed (which informs downstream invalidation).
        """

        artifact = (
            self.session.query(StageArtifact)
            .filter(StageArtifact.stage_name == stage_name, StageArtifact.item_key == item_key)
            .one_or_none()
        )
        changed = False
        if artifact is None:
            artifact = StageArtifact(
                stage_name=stage_name,
                item_key=item_key,
                input_hash=input_hash,
                payload=payload,
            )
            artifact.mark_updated()
            self.session.add(artifact)
            changed = True
        else:
            payload_changed = artifact.payload != payload
            hash_changed = artifact.input_hash != input_hash
            needs_update = payload_changed or hash_changed or artifact.needs_rerun
            if needs_update:
                artifact.input_hash = input_hash
                artifact.payload = payload
                artifact.mark_updated()
                changed = True
            else:
                artifact.mark_updated()
        return artifact, changed

    def mark_dependents_dirty(self, stage_name: str, dependent_stage_names: Set[str]) -> None:
        if not dependent_stage_names:
            return
        for dependent in dependent_stage_names:
            (
                self.session.query(StageArtifact)
                .filter(StageArtifact.stage_name == dependent)
                .update(
                    {
                        StageArtifact.needs_rerun: True,
                        StageArtifact.status: "stale",
                        StageArtifact.updated_at: datetime.datetime.utcnow(),
                    },
                    synchronize_session=False,
                )
            )


class PipelineContext:
    """Holds shared execution state for a pipeline run."""

    def __init__(self, session: Session):
        self.session = session
        self.store = StageStateStore(session)


class StageContext:
    """Context passed to each stage during execution."""

    def __init__(self, pipeline_context: PipelineContext, task: "Task") -> None:
        self._pipeline_context = pipeline_context
        self.task = task
        self.skipped_items = 0
        self._logger = logging.getLogger(f"{__name__}.{task.name}")

    @property
    def session(self) -> Session:
        return self._pipeline_context.session

    @property
    def store(self) -> StageStateStore:
        return self._pipeline_context.store

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def record_skip(self) -> None:
        self.skipped_items += 1


class Task:
    """Base class for individual pipeline stages."""

    def __init__(self, name: str, *, depends_on: Optional[Sequence[str]] = None, description: str = "") -> None:
        self.name = name
        self.depends_on: Set[str] = set(depends_on or [])
        self.description = description

    # -- Hooks to override ------------------------------------------------
    def on_stage_started(self, context: StageContext) -> None:  # pragma: no cover - hook
        """Hook called before any work items are processed."""

    def produce_items(self, context: StageContext) -> Iterable[Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def item_key(self, context: StageContext, item: Any) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def item_input_hash(self, context: StageContext, item: Any) -> str:
        return stable_hash(item)

    def process(self, context: StageContext, work_item: WorkItem) -> Any:
        """Perform the work for *work_item* and return the result payload."""

        return work_item.data

    def result_payload(self, context: StageContext, work_item: WorkItem, result: Any) -> Any:
        return result

    def on_skip(self, context: StageContext, artifact: StageArtifact) -> None:  # pragma: no cover - hook
        """Hook called when an item is skipped because it is already up to date."""

    def on_stage_completed(self, context: StageContext) -> None:  # pragma: no cover - hook
        """Hook called after all items have been processed (or skipped)."""

    # -- Helper methods ---------------------------------------------------
    def iter_work_items(self, context: StageContext) -> Iterator[WorkItem]:
        for item in self.produce_items(context):
            key = self.item_key(context, item)
            if key is None:
                continue
            input_hash = self.item_input_hash(context, item)
            artifact = context.store.get_artifact(self.name, key)
            if artifact and artifact.input_hash == input_hash and not artifact.needs_rerun:
                context.record_skip()
                context.store.mark_artifact_clean(artifact)
                self.on_skip(context, artifact)
                continue
            yield WorkItem(key=key, data=item, input_hash=input_hash, artifact=artifact)


class Pipeline:
    """Execute tasks while respecting dependencies and cached artifacts."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory
        self._tasks: Dict[str, Task] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)

    def register(self, task: Task) -> None:
        if task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already registered")
        self._tasks[task.name] = task
        self._dependencies[task.name] = set(task.depends_on)
        for dep in task.depends_on:
            self._dependents[dep].add(task.name)

    # -- Execution --------------------------------------------------------
    def run(self, *, selected_stages: Optional[Sequence[str]] = None, note: Optional[str] = None) -> None:
        order = self._execution_order(selected_stages)
        if not order:
            logger.info("No stages selected for execution")
            return

        with self._session_factory() as session:
            context = PipelineContext(session)
            run_record = context.store.start_run(note=note)
            try:
                for stage_name in order:
                    task = self._tasks[stage_name]
                    stage_context = StageContext(context, task)
                    logger.info("Starting stage %s", stage_name)
                    task.on_stage_started(stage_context)
                    stage_run = context.store.start_stage_run(run_record, stage_name)
                    processed_items = 0
                    changed_any = False
                    try:
                        for work_item in task.iter_work_items(stage_context):
                            result = task.process(stage_context, work_item)
                            payload = task.result_payload(stage_context, work_item, result)
                            artifact, changed = context.store.record_artifact(
                                stage_name, work_item.key, work_item.input_hash, payload
                            )
                            if changed:
                                changed_any = True
                                context.store.mark_dependents_dirty(stage_name, self._dependents.get(stage_name, set()))
                            processed_items += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Stage %s failed", stage_name)
                        context.store.fail_stage_run(stage_run, str(exc))
                        context.store.fail_run(run_record, message=f"Stage '{stage_name}' failed")
                        session.rollback()
                        raise
                    else:
                        task.on_stage_completed(stage_context)
                        status = "completed" if processed_items > 0 or changed_any else "skipped"
                        context.store.finish_stage_run(
                            stage_run,
                            status=status,
                            processed_items=processed_items,
                            skipped_items=stage_context.skipped_items,
                        )
                        session.commit()
                        logger.info(
                            "Finished stage %s (processed=%s skipped=%s status=%s)",
                            stage_name,
                            processed_items,
                            stage_context.skipped_items,
                            status,
                        )
            except Exception:
                logger.exception("Pipeline run failed")
                context.store.fail_run(run_record, message="Pipeline execution failed")
                session.commit()
                raise
            else:
                context.store.finish_run(run_record, status="completed")
                session.commit()
                logger.info("Pipeline run %s completed", run_record.id)

    # -- Helpers ----------------------------------------------------------
    def _execution_order(self, selected: Optional[Sequence[str]]) -> List[str]:
        if selected is None:
            selected_set = set(self._tasks.keys())
        else:
            selected_set = self._expand_with_dependencies(set(selected))

        return self._topological_sort(selected_set)

    def _expand_with_dependencies(self, initial: Set[str]) -> Set[str]:
        to_process = list(initial)
        seen = set(initial)
        while to_process:
            name = to_process.pop()
            for dep in self._dependencies.get(name, set()):
                if dep not in seen:
                    seen.add(dep)
                    to_process.append(dep)
        return seen

    def _topological_sort(self, stage_names: Set[str]) -> List[str]:
        temp_mark: Set[str] = set()
        perm_mark: Set[str] = set()
        result: List[str] = []

        def visit(node: str) -> None:
            if node in perm_mark:
                return
            if node in temp_mark:
                raise ValueError("Circular dependency detected in pipeline tasks")
            temp_mark.add(node)
            for dep in self._dependencies.get(node, set()):
                if dep in stage_names:
                    visit(dep)
            temp_mark.remove(node)
            perm_mark.add(node)
            if node in stage_names:
                result.append(node)

        for name in sorted(stage_names):
            visit(name)
        return result
