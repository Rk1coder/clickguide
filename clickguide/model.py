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
                "Locate the UI target described below in this screenshot. Return ONLY a JSON "
                "array [x, y] for its center point. Coordinates are normalized to [0, 1], "
                "where (0,0) is the top-left and (1,1) is the bottom-right of the entire image. "
                "Do not return a bounding box or explanation. Target: " + instruction
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
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            generated = output[:, inputs.input_ids.shape[1] :]
            return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def create_engine(name):
    if name == "showui":
        return ShowUIEngine()
    if name == "qwen":
        return QwenEngine()
    raise ValueError("CLICKGUIDE_MODEL must be 'showui' or 'qwen'.")
