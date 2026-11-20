from pathlib import Path

from PIL import Image

from app import preview_upload, run_locate
from clickguide.service import GuideService


class Engine:
    model_id = "test"
    revision = "test"

    def predict(self, image, instruction):
        return "[0.5, 0.5]"


def test_callback_returns_image_and_report(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (50, 50)).save(image)
    service = GuideService(Engine(), tmp_path / "reports")
    output = run_locate(str(image), "Open settings", service=service)
    assert Path(output[0]).is_file()
    assert Path(output[1]).is_file()
    assert "25" in output[2] and "suggestion" in output[2].lower()
    assert Path(output[3]).suffix == ".json"
    assert output[4] == output[0]


def test_validation_failure_clears_previous_outputs(tmp_path):
    service = GuideService(Engine(), tmp_path / "reports")
    for path, query in [(None, "Open settings"), ("missing.png", "ab")]:
        output = run_locate(path, query, service=service)
        assert output[0] is None and output[1] is None
        assert output[3] is None and output[4] is None
        assert output[2]


def test_backend_failure_is_clear_without_exposing_internal_details(tmp_path):
    class BrokenEngine(Engine):
        def predict(self, image, instruction):
            raise RuntimeError("private backend details")

    image = tmp_path / "image.png"
    Image.new("RGB", (50, 50)).save(image)
    service = GuideService(BrokenEngine(), tmp_path / "reports")
    output = run_locate(str(image), "Open settings", service=service)
    assert output[0] is None and output[1] is None and output[3] is None and output[4] is None
    assert "private backend details" not in output[2]
    assert "failed" in output[2].lower()


def test_uploaded_image_preview_uses_same_validation(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (50, 30), "blue").save(image)
    assert preview_upload(str(image)).size == (50, 30)
    assert preview_upload(None) is None
    image.write_bytes(b"invalid")
    assert preview_upload(str(image)) is None
