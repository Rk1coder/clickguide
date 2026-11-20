import logging
import os
import threading
import uuid
from pathlib import Path

from clickguide.core import load_image
from clickguide.model import create_engine
from clickguide.service import GuideService
from clickguide.storage import private_cache, prune_cache

CACHE = private_cache(os.environ.get("CLICKGUIDE_CACHE_DIR"))
prune_cache(CACHE)
os.environ["GRADIO_TEMP_DIR"] = str(CACHE)

# Gradio must discover the private cache after its environment has been configured.
import gradio as gr  # noqa: E402

ROOT = Path(__file__).resolve().parent
SERVICE = GuideService(create_engine(os.environ.get("CLICKGUIDE_MODEL", "qwen")))
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


def invalidate_inputs(path, instruction, session):
    session["token"] = uuid.uuid4().hex
    session["inputs"] = (path, instruction)
    return None, None, "Inputs changed. Locate the target to get a new suggestion.", None, None


def _on_upload(path, instruction, session):
    cleared = invalidate_inputs(path, instruction, session)
    return preview_upload(path), *cleared


def _controls(interactive):
    return [gr.update(interactive=interactive) for _ in range(3)]


def run_locate(path, instruction, session=None, *, service=None):
    service = service or SERVICE
    ticket = session["token"] if session is not None else None
    if session is not None and session.get("inputs") not in (None, (path, instruction)):
        return (gr.skip(),) * 5
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
        output = (
            result["image_path"],
            result["crop_path"],
            status,
            result["report_path"],
            result["image_path"],
        )
    except ValueError as exc:
        output = None, None, str(exc), None, None
    except Exception:
        LOGGER.exception("Local grounding failed")
        output = (
            None,
            None,
            "Local inference failed. Check the server log and available memory, then retry.",
            None,
            None,
        )
    if session is not None and session["token"] != ticket:
        return (gr.skip(),) * 5
    return output


def build_app():
    sample_paths = [ROOT / "examples" / f"{scene}.jpg" for scene in ("settings", "board", "files")]
    gr.set_static_paths(paths=sample_paths)
    with gr.Blocks(title="ClickGuide", delete_cache=(60, 3600)) as demo:
        session = gr.State(value={"token": "initial", "inputs": None})
        gr.Markdown(
            "# ClickGuide\nFind your next click.\n\n"
            "Upload a screenshot and describe one target in English. A local model suggests a location "
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
                        [str(sample_paths[0]), "Save changes"],
                        [str(sample_paths[1]), "Create a new task"],
                        [str(sample_paths[2]), "Upload a file"],
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
        outputs = [marked, crop, status, report_file, marked_file]
        request = locate.click(
            lambda: _controls(False), outputs=[upload, query, locate], queue=False, api_name=False
        )
        prediction = request.then(
            run_locate,
            [upload, query, session],
            outputs,
            api_name="locate",
            concurrency_limit=1,
            concurrency_id="model",
        )
        prediction.then(
            lambda: _controls(True), outputs=[upload, query, locate], queue=False, api_name=False
        )
        upload.change(
            _on_upload,
            [upload, query, session],
            [original, *outputs],
            api_name=False,
            concurrency_limit=2,
        )
        query.change(
            invalidate_inputs, [upload, query, session], outputs, queue=False, api_name=False
        )
        gr.Markdown(
            f"**Local-first preview** · {SERVICE.engine.model_id.split('/')[-1]} · English instructions · Temporary outputs expire after one hour. Original examples contain fictional data."
        )
    return demo.queue(max_size=4, default_concurrency_limit=1)


def _cleanup_loop():
    while not threading.Event().wait(60):
        SERVICE.prune()
        prune_cache(CACHE)


if __name__ == "__main__":
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    build_app().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7864")),
        max_file_size="10mb",
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
