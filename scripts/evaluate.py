"""Measure the pinned model on every original development task."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickguide.model import MAX_PIXELS, MIN_PIXELS, create_engine  # noqa: E402
from clickguide.service import GuideService, score_point  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation/local-cpu.json"))
    parser.add_argument("--model", choices=["showui", "qwen"], default="showui")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "examples/tasks.json").read_text())
    service = GuideService(create_engine(args.model))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images = root / "docs/images"
    images.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        row = {"id": task["id"], "target_box": task["target_box"]}
        try:
            result = service.locate(root / "examples" / task["image"], task["instruction"])
            row["report"] = result["report"]
            row["hit"] = score_point(result["report"]["pixel_point"], task["target_box"])
            prefix = "qwen-" if args.model == "qwen" else ""
            shutil.copyfile(result["image_path"], images / f"{prefix}{task['id']}.png")
        except Exception as exc:
            row.update(hit=False, error=f"{type(exc).__name__}: {exc}")
            if hasattr(exc, "raw_prediction"):
                row["raw_prediction"] = exc.raw_prediction
        results.append(row)
        record = {
            "scope": "Original fictional development tasks; not an independent benchmark.",
            "settings": {
                "min_pixels": MIN_PIXELS if args.model == "showui" else 65536,
                "max_pixels": MAX_PIXELS if args.model == "showui" else 262144,
                "max_new_tokens": 32,
                "device": "cpu",
                "dtype": "bfloat16" if args.model == "showui" else "float32",
                "image_processor": "fast",
                "threads": max(1, min(16, int(os.environ.get("CLICKGUIDE_THREADS", "4")))),
            },
            "results": results,
            "hits": sum(r["hit"] for r in results),
            "errors": sum("error" in r for r in results),
            "total": len(results),
        }
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(
            task["id"],
            "HIT" if row["hit"] else "MISS",
            row.get("error", row.get("report", {}).get("raw_prediction")),
            flush=True,
        )
    return 1 if any("error" in row for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
