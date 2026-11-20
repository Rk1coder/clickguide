import json
import os
import threading

MODEL_ID = "showlab/ShowUI-2B"
REVISION = "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
SYSTEM = (
    "Based on the screenshot of the page, I give a text description and you give its "
    "corresponding location. The coordinate represents a clickable location [x, y] for "
    "an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
)


class ShowUIEngine:
    model_id = MODEL_ID
    revision = REVISION

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def predict(self, image, instruction):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        with self._lock:
            if self._model is None:
                torch.set_num_threads(
                    max(1, min(16, int(os.environ.get("CLICKGUIDE_THREADS", "4"))))
                )
                self._processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    revision=REVISION,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS,
                    use_fast=True,
                    trust_remote_code=False,
                )
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    revision=REVISION,
                    dtype=torch.bfloat16,
                    device_map="cpu",
                    attn_implementation="sdpa",
                    trust_remote_code=False,
                    weights_only=True,
                ).eval()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM},
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            generated = output[:, inputs.input_ids.shape[1] :]
            return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class QwenEngine:
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    revision = "89644892e4d85e24eaac8bacfd4f463576704203"
    coordinate_scale = 1000
    parse_prediction = staticmethod(lambda raw: parse_qwen_point(raw))

    def __init__(self):
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def predict(self, image, instruction):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        with self._lock:
            if self._model is None:
                torch.set_num_threads(
                    max(1, min(16, int(os.environ.get("CLICKGUIDE_THREADS", "4"))))
                )
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    min_pixels=65536,
                    max_pixels=262144,
                    use_fast=True,
                    trust_remote_code=False,
                )
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    dtype=torch.float32,
                    device_map="cpu",
                    attn_implementation="sdpa",
                    trust_remote_code=False,
                    weights_only=True,
                ).eval()
            prompt = (
                "Point to the following UI target: "
                + instruction
                + '. Output its center as JSON {"point_2d": [x, y]} using integer '
                "coordinates from 0 to 1000 relative to the entire screenshot."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            generated = output[:, inputs.input_ids.shape[1] :]
            return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_qwen_point(raw):
    if not isinstance(raw, str) or len(raw) > 512:
        raise ValueError("Model did not return a valid native point.")
    raw = raw.strip()
    if raw.startswith("```json\n") and raw.endswith("\n```"):
        raw = raw[8:-4]
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Model did not return a valid native point.") from exc
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("point_2d")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(v) is not int or not 0 <= v <= 1000 for v in value)
    ):
        raise ValueError("Expected one integer point in the native 0–1000 coordinate range.")
    return value[0] / 1000, value[1] / 1000


def create_engine(name):
    if name == "showui":
        return ShowUIEngine()
    if name == "qwen":
        return QwenEngine()
    raise ValueError("CLICKGUIDE_MODEL must be 'showui' or 'qwen'.")
