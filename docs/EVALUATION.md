# Development evaluation

Six tasks across three original fictional interfaces. This is development data, not an independent benchmark. The same six task instructions and target regions were retained across both runs. No target annotations are passed to a model.

## Results

| Task | ShowUI point | Hit | Qwen point | Hit |
| --- | --- | --- | --- | --- |
| settings-notifications | [100, 224] | No | [275, 203] | No |
| settings-save | [850, 563] | Yes | [851, 559] | Yes |
| board-create | [850, 115] | Yes | [861, 122] | Yes |
| board-onboarding | [330, 218] | No | [321, 291] | Yes |
| files-upload | [840, 38] | Yes | [874, 37] | Yes |
| files-download | [870, 256] | No | [888, 297] | Yes |

ShowUI: **3/6**. Qwen3-VL: **5/6**. Qwen still misses the Notifications task: it points into the main content rather than the sidebar target. The instruction is potentially ambiguous because the screen already displays notification settings; the original target annotation remains unchanged.

Point-in-box uses inclusive bounds of the predefined target. Coordinates map to the EXIF-oriented RGB image. At the right/bottom image boundary, normalized 1.0 is clamped to the final zero-based pixel. These results neither establish absent-target detection nor imply that every valid coordinate is a correct target.

## Configuration

| Setting | ShowUI | Qwen3-VL |
| --- | --- | --- |
| Model | showlab/ShowUI-2B | Qwen/Qwen3-VL-2B-Instruct |
| CPU dtype | bfloat16 | float32 |
| Image pixel budget | 200704–401408 | 65536–262144 |
| Generated token cap | 32 | 96 |
| Raw coordinate scale | 1 | 1000 |
| Processor | Fast | Fast |
| CPU threads | 4 | 4 |
| Decoding | Greedy | Greedy |

The defaults differ by backend. Latency and hit counts compare these configured pipelines and do not isolate model architecture. Qwen was selected after examining these development examples, so the six cases must not be described as a held-out test. Both adapters use immutable revisions recorded in every report.

## Timing

Times cover inference and any first-call model/processor loading; they exclude service image exports. First calls are cold model loads with weights already cached. Network downloads are excluded.

- ShowUI: 74.934–88.926 seconds per task; first call 88.926 seconds.
- Qwen3-VL: 9.422–16.623 seconds per task; first call 16.623 seconds.

## Reproduce

Install the documented Python 3.12 environment and run:

```bash
python scripts/evaluate.py --model showui --output evaluation/local-cpu.json
python scripts/evaluate.py --model qwen --output evaluation/qwen-cpu.json
```

The runner records every task, writes JSON after each task, and exits nonzero when a processing/format error occurs. Such errors are reported separately from valid predictions that miss their target. Source images and fixed boxes are in `examples/`; actual annotated PNGs are in `docs/images/`.

Fixtures were rendered from `examples/fixtures.html` in a browser and captured at 1000 × 640 pixels. Target boxes came from the fixture DOM geometry before inference, except the onboarding card uses its complete card bounds. This DOM information is used only for evaluation.

## Known limitations

Only six tasks, English labels, a single desktop resolution, fictional products, and visible targets are covered. Mobile screens, high DPI, inaccessible targets, absent/ambiguous elements, adversarial screenshot text, and real application workflows require further evaluation. Model suggestions never trigger clicks.

## Next experiments

- Freeze a separate, independently annotated evaluation set before further prompt changes.
- Add absent-target and ambiguous-target tasks with explicit failure metrics.
- Evaluate higher-resolution input and crop refinement against the fixed baseline.
- Measure memory use and compare dtypes without changing other settings.
