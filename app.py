import logging
import os
import threading
from pathlib import Path

import gradio as gr

from clickguide.core import load_image
from clickguide.model import ShowUIEngine
from clickguide.service import GuideService

ROOT = Path(__file__).resolve().parent
SERVICE = GuideService(ShowUIEngine())
LOGGER = logging.getLogger(__name__)

CSS = """
.gradio-container{max-width:1440px!important;font-family:system-ui,sans-serif!important}
#masthead{border-bottom:1px solid #dbe2ea;padding:12px 0 24px;margin-bottom:20px}
#masthead h1{font-size:32px;letter-spacing:-.02em;color:#172033;margin-bottom:10px}
#masthead p{max-width:72ch;line-height:1.6;color:#506079}
#instruction-panel{background:#f6f8fb;border:1px solid #dbe2ea;border-radius:10px;padding:20px}
#result-panel{padding:0 8px}
#locate-button{min-height:48px;font-weight:600;letter-spacing:.01em}
#result-status{padding:16px;background:#f6f8fb;border:1px solid #dbe2ea;border-radius:8px;line-height:1.6}
button:focus-visible,a:focus-visible{outline:3px solid #2563eb!important;outline-offset:3px}
@media(max-width:720px){#masthead h1{font-size:28px}#instruction-panel{padding:12px}}
"""


def preview_upload(path):
    if not path:
        return None
    try:
        return load_image(path)
    except ValueError:
        return None


def run_locate(path, instruction, *, service=None):
    service = service or SERVICE
    if not path:
        return None, None, "Upload a screenshot to begin.", None, None
    try:
        result = service.locate(path, instruction)
        report = result["report"]
        x, y = report["pixel_point"]
        w, h = report["image_size"]
        status = (
            f"**Model suggestion · ({x}, {y}) px**\n\n"
            f"Image: {w} × {h} px · {report['elapsed_seconds']:.1f} s including any model loading.\n\n"
            "Inspect the red crosshair before acting. This model can suggest a point even "
            "when the requested target is absent or ambiguous."
        )
        return (
            result["image_path"],
            result["crop_path"],
            status,
            result["report_path"],
            result["image_path"],
        )
    except ValueError as exc:
        return None, None, str(exc), None, None
    except Exception:
        LOGGER.exception("Local grounding failed")
        return (
            None,
            None,
            "Local inference failed. Check the server log and available memory, then retry.",
            None,
            None,
        )


def build_app():
    sample_paths = [ROOT / "examples" / f"{scene}.jpg" for scene in ("settings", "board", "files")]
    gr.set_static_paths(paths=sample_paths)
    with gr.Blocks(title="ClickGuide", delete_cache=(60, 3600)) as demo:
        gr.Markdown(
            "# ClickGuide\nFind your next click.\n\n"
            "Upload a screenshot and describe one target in English. ShowUI suggests a location "
            "you can inspect. Runs on this machine; the application does not control your desktop.",
            elem_id="masthead",
        )
        with gr.Row():
            with gr.Column(scale=4, min_width=300, elem_id="instruction-panel"):
                gr.Markdown("### 01 / Choose a screenshot")
                upload = gr.File(
                    label="Screenshot · PNG or JPEG · up to 10 MiB",
                    file_types=[".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                original = gr.Image(
                    label="Original screenshot",
                    interactive=False,
                    height=220,
                    buttons=["download", "fullscreen"],
                    format="png",
                )
                query = gr.Textbox(
                    label="What would you like to click?",
                    placeholder="Open Notifications settings",
                    lines=2,
                    max_lines=4,
                )
                locate = gr.Button("Locate target", variant="primary", elem_id="locate-button")
                gr.Markdown(
                    "The first request downloads and loads the model. CPU inference may take a while. Keep the target visible in the screenshot."
                )
                gr.Examples(
                    examples=[
                        [str(sample_paths[0]), "Open Notifications settings"],
                        [str(sample_paths[1]), "Create a new task"],
                        [str(sample_paths[2]), "Download the quarterly report"],
                    ],
                    inputs=[upload, query],
                    label="Try an original example",
                    cache_examples=False,
                )
            with gr.Column(scale=7, min_width=300, elem_id="result-panel"):
                gr.Markdown("### 02 / Inspect the suggestion")
                marked = gr.Image(
                    label="Suggested target",
                    type="filepath",
                    interactive=False,
                    height=420,
                    buttons=["download", "fullscreen"],
                    format="png",
                )
                status = gr.Markdown(
                    "Your marked screenshot will appear here. No model has run yet.",
                    elem_id="result-status",
                )
                with gr.Accordion("Magnified target", open=False):
                    crop = gr.Image(
                        label="Target detail",
                        type="filepath",
                        interactive=False,
                        height=280,
                        buttons=["download", "fullscreen"],
                        format="png",
                    )
                with gr.Row():
                    report_file = gr.File(label="Download JSON report", interactive=False)
                    marked_file = gr.File(label="Download annotated PNG", interactive=False)
        locate.click(
            run_locate,
            [upload, query],
            [marked, crop, status, report_file, marked_file],
            api_name="locate",
            concurrency_limit=1,
            concurrency_id="model",
        )
        upload.change(preview_upload, upload, original, api_name="preview", concurrency_limit=2)
        gr.Markdown(
            "**Local-first preview** · ShowUI-2B · English instructions · Temporary outputs expire after one hour. Original examples contain fictional data."
        )
    return demo.queue(max_size=4, default_concurrency_limit=1)


def _cleanup_loop():
    while not threading.Event().wait(60):
        SERVICE.prune()


if __name__ == "__main__":
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    build_app().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7864")),
        max_file_size="10mb",
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
