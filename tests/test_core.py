import math

import pytest
from PIL import Image

from clickguide.core import (
    annotate,
    load_image,
    parse_point,
    pixel_point,
    target_crop,
    validate_instruction,
)


@pytest.mark.parametrize("raw, expected", [("[0.25, 0.75]", (0.25, 0.75)), ("(0, 1)", (0.0, 1.0))])
def test_parse_coordinate_pair(raw, expected):
    assert parse_point(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "[true, 0.5]",
        "[True, 0.5]",
        "[NaN, 0]",
        "[1e999, 0]",
        "[-0.1, 0.2]",
        "[1.01, 0]",
        "[0.2]",
        "[[0.2, 0.3]]",
        '["0.2", 0.3]',
        "[0.2, 0.3, 0.4]",
        '__import__("os").system("false")',
        "[0.2, 0.3] extra",
    ],
)
def test_reject_invalid_predictions(raw):
    with pytest.raises(ValueError):
        parse_point(raw)


@pytest.mark.parametrize(
    "point, expected", [((0, 0), (0, 0)), ((1, 1), (199, 99)), ((0.5, 0.5), (100, 50))]
)
def test_pixel_coordinates_stay_inside_image(point, expected):
    assert pixel_point(point, (200, 100)) == expected


@pytest.mark.parametrize("point", [(math.nan, 0), (0, math.inf), (-1, 0), (True, 0)])
def test_direct_invalid_points_are_rejected(point):
    with pytest.raises(ValueError):
        pixel_point(point, (200, 100))


def test_marker_matches_pixel_and_preserves_input():
    image = Image.new("RGB", (200, 100), "white")
    marked = annotate(image, (0.5, 0.5))
    assert image.getpixel((100, 50)) == (255, 255, 255)
    assert marked.getpixel((100, 50)) == (220, 38, 38)
    assert marked.size == (200, 100)
    assert target_crop(image, (1, 1)).size == (100, 100)


def test_normalize_alpha_and_orientation(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(path)
    assert load_image(path).getpixel((0, 0)) == (255, 255, 255)
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (20, 10), "blue")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert load_image(path).size == (10, 20)


def test_reject_corrupt_and_unsupported_images(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(bad)
    gif = tmp_path / "animated.gif"
    Image.new("RGB", (10, 10)).save(
        gif, save_all=True, append_images=[Image.new("RGB", (10, 10), "red")]
    )
    with pytest.raises(ValueError):
        load_image(gif)


def test_reject_oversized_dimension_before_decode(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (4097, 1)).save(path)
    with pytest.raises(ValueError):
        load_image(path)


def test_reject_large_file_before_open(tmp_path):
    path = tmp_path / "large.png"
    with path.open("wb") as f:
        f.truncate(10 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        load_image(path)


def test_instruction_normalization_and_bounds():
    assert validate_instruction("  Open settings  ") == "Open settings"
    for value in ("", "ab", "x" * 301, None):
        with pytest.raises(ValueError):
            validate_instruction(value)
