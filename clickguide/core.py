import ast
import math
import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

Point = tuple[float, float]


def validate_instruction(text: str) -> str:
    if not isinstance(text, str) or not 3 <= len(text.strip()) <= 300:
        raise ValueError("Enter an English instruction of 3–300 characters.")
    return text.strip()


def load_image(path: str | Path) -> Image.Image:
    try:
        path = Path(path)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Screenshot exceeds the 10 MiB limit.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.format not in {"PNG", "JPEG"} or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Upload one non-animated PNG or JPEG screenshot.")
                w, h = source.size
                if max(w, h) > 4096 or w * h > 12_000_000:
                    raise ValueError(
                        "Screenshot must fit 4096 pixels per edge and 12 million pixels."
                    )
                image = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                return Image.alpha_composite(background, image).convert("RGB")
    except (
        OSError,
        TypeError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Cannot read this screenshot. Upload a valid PNG or JPEG.") from exc


def parse_point(raw: str) -> Point:
    if not isinstance(raw, str) or len(raw) > 200:
        raise ValueError("Model did not return a valid coordinate pair.")
    try:
        value = ast.literal_eval(raw.strip())
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Model did not return a valid coordinate pair.") from exc
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Model did not return exactly two coordinates.")
    _validate_point(value)
    return float(value[0]), float(value[1])


def _validate_point(point):
    if len(point) != 2 or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in point
    ):
        raise ValueError("Coordinates must be finite numbers from 0 to 1.")


def pixel_point(point: Point, size: tuple[int, int]) -> tuple[int, int]:
    _validate_point(point)
    w, h = size
    if w < 1 or h < 1:
        raise ValueError("Image dimensions must be positive.")
    return min(w - 1, round(point[0] * w)), min(h - 1, round(point[1] * h))


def annotate(image: Image.Image, point: Point) -> Image.Image:
    marked = image.copy()
    x, y = pixel_point(point, image.size)
    draw = ImageDraw.Draw(marked)
    radius = max(10, min(image.size) // 45)
    draw.ellipse(
        (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), outline="white", width=5
    )
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(220, 38, 38), width=3)
    draw.line((x - radius - 6, y, x + radius + 6, y), fill=(220, 38, 38), width=2)
    draw.line((x, y - radius - 6, x, y + radius + 6), fill=(220, 38, 38), width=2)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 38, 38))
    return marked


def target_crop(image: Image.Image, point: Point) -> Image.Image:
    x, y = pixel_point(point, image.size)
    w, h = image.size
    side = min(320, w, h)
    left = min(max(0, x - side // 2), w - side)
    top = min(max(0, y - side // 2), h - side)
    return annotate(image, point).crop((left, top, left + side, top + side))
