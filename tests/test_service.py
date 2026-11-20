import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from clickguide.service import GuideService, score_point


class FixedEngine:
    model_id = "test-fixture"
    revision = "test-only"

    def __init__(self, raw="[0.5, 0.5]"):
        self.raw = raw

    def predict(self, image, instruction):
        assert image.mode == "RGB"
        assert instruction == "Open settings"
        return self.raw


@pytest.fixture
def screenshot(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def test_complete_report_and_images(tmp_path, screenshot):
    service = GuideService(FixedEngine(), tmp_path / "reports")
    result = service.locate(screenshot, " Open settings ")
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["pixel_point"] == [100, 50]
    assert report["normalized_point"] == [0.5, 0.5]
    assert report["image_size"] == [200, 100]
    assert report["model"] == {"id": "test-fixture", "revision": "test-only"}
    assert len(report["image_sha256"]) == 64
    assert report["status"] == "suggested"
    assert Image.open(result["image_path"]).getpixel((100, 50)) == (220, 38, 38)
    assert Path(result["crop_path"]).is_file()


def test_parallel_results_do_not_overwrite(tmp_path, screenshot):
    service = GuideService(FixedEngine(), tmp_path / "reports")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: service.locate(screenshot, "Open settings"), range(2)))
    assert results[0]["report_path"] != results[1]["report_path"]
    assert all(Path(r["report_path"]).is_file() for r in results)


def test_parallel_requests_serialize_model_work(tmp_path, screenshot):
    class BusyEngine(FixedEngine):
        active = 0
        peak = 0

        def predict(self, image, instruction):
            self.active += 1
            self.peak = max(self.peak, self.active)
            time.sleep(0.02)
            self.active -= 1
            return "[0.5, 0.5]"

    engine = BusyEngine()
    service = GuideService(engine, tmp_path / "reports")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: service.locate(screenshot, "Open settings"), range(3)))
    assert engine.peak == 1


@pytest.mark.parametrize("raw", ["not coordinates", "[2,0]", "[True,0]"])
def test_invalid_prediction_produces_no_export(tmp_path, screenshot, raw):
    service = GuideService(FixedEngine(raw), tmp_path / "reports")
    with pytest.raises(ValueError):
        service.locate(screenshot, "Open settings")
    assert list((tmp_path / "reports").iterdir()) == []


def test_invalid_prediction_retains_raw_output_for_evaluation(tmp_path, screenshot):
    service = GuideService(FixedEngine("[850, 120]"), tmp_path / "reports")
    with pytest.raises(ValueError) as exc:
        service.locate(screenshot, "Open settings")
    assert exc.value.raw_prediction == "[850, 120]"


def test_expiration_and_report_count(tmp_path, screenshot):
    service = GuideService(FixedEngine(), tmp_path / "reports", max_reports=2)
    first = service.locate(screenshot, "Open settings")
    expired = Path(first["report_path"]).parent
    os.utime(expired, (time.time() - 4000, time.time() - 4000))
    second = service.locate(screenshot, "Open settings")
    assert not expired.exists()
    third = service.locate(screenshot, "Open settings")
    fourth = service.locate(screenshot, "Open settings")
    assert not Path(second["report_path"]).exists()
    assert Path(third["report_path"]).exists()
    assert Path(fourth["report_path"]).exists()
    assert screenshot.exists()


@pytest.mark.parametrize(
    "point, expected", [((10, 20), True), ((30, 40), True), ((9, 30), False), ((20, 41), False)]
)
def test_point_in_target_uses_declared_bounds(point, expected):
    assert score_point(point, [10, 20, 30, 40]) is expected
