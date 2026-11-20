import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from clickguide.core import (
    annotate,
    load_image,
    parse_point,
    pixel_point,
    target_crop,
    validate_instruction,
)


def score_point(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


class PredictionError(ValueError):
    def __init__(self, message, raw_prediction):
        super().__init__(message)
        self.raw_prediction = raw_prediction


class GuideService:
    def __init__(self, engine, root=None, max_reports=32):
        self.engine = engine
        self.root = (
            Path(root) if root else Path(tempfile.gettempdir()) / f"clickguide-{os.getuid()}"
        )
        if self.root.is_symlink():
            raise ValueError("Report directory must not be a symlink.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.stat().st_uid != os.getuid():
            raise ValueError("Report directory must belong to the current user.")
        self.root.chmod(0o700)
        self.max_reports = max_reports
        self._lock = threading.RLock()
        self.prune()

    def prune(self, reserve=0):
        with self._lock:
            reports = sorted(
                (p for p in self.root.glob("cg-*") if p.is_dir() and not p.is_symlink()),
                key=lambda p: p.stat().st_mtime,
            )
            fresh = []
            for p in reports:
                if time.time() - p.stat().st_mtime > 3600:
                    shutil.rmtree(p)
                else:
                    fresh.append(p)
            for p in fresh[: max(0, len(fresh) - max(0, self.max_reports - reserve))]:
                shutil.rmtree(p)

    def locate(self, path, instruction):
        instruction = validate_instruction(instruction)
        image = load_image(path)
        with self._lock:
            started = time.perf_counter()
            raw = self.engine.predict(image, instruction)
            try:
                point = getattr(self.engine, "parse_prediction", parse_point)(raw)
            except ValueError as exc:
                raise PredictionError(str(exc), raw) from exc
            elapsed = time.perf_counter() - started
            pixel = pixel_point(point, image.size)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            report = {
                "schema_version": 1,
                "status": "suggested",
                "instruction": instruction,
                "normalized_point": list(point),
                "pixel_point": list(pixel),
                "image_size": list(image.size),
                "coordinate_frame": "EXIF-oriented RGB screenshot; zero-based pixel coordinates",
                "image_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
                "model": {"id": self.engine.model_id, "revision": self.engine.revision},
                "raw_prediction": raw,
                "raw_coordinate_scale": getattr(self.engine, "coordinate_scale", 1),
                "elapsed_seconds": round(elapsed, 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "notice": "Model suggestion; coordinate validity does not establish target correctness.",
            }
            self.prune(reserve=1)
            directory = Path(tempfile.mkdtemp(prefix="cg-", dir=self.root))
            try:
                annotate(image, point).save(directory / "target.png")
                target_crop(image, point).save(directory / "detail.png")
                (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            except Exception:
                shutil.rmtree(directory)
                raise
            return {
                "report": report,
                "report_path": str(directory / "report.json"),
                "image_path": str(directory / "target.png"),
                "crop_path": str(directory / "detail.png"),
            }
