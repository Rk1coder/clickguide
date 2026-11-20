# ClickGuide: local-first design

## Intent

Turn an English instruction and a screenshot into a visibly marked suggested click
location. Build and test locally, retain meaningful stage commits, then publish
source and non-sensitive measured output images on GitHub. Public copy is English.

## Approaches

1. ShowUI-2B screenshot grounding: chosen starting point because the model is trained
   for coordinate prediction. CPU inference will be measured, not called real time.
2. General Qwen3-VL: useful future comparison, outside the first release.
3. Fully autonomous desktop control: outside scope; users inspect a suggestion.

## First release

- Python 3.12, PyTorch CPU, Transformers, Pillow, Gradio; separate `develop` repo.
- Model `showlab/ShowUI-2B`, immutable revision
  `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`. No remote Python model code.
- English instruction, 3–300 characters; one PNG/JPEG, <=10 MiB, <=12 million
  pixels, <=4096 pixels per edge; reject animation and malformed images.
- Correct EXIF orientation before inference; alpha composited over white. Report
  dimensions and coordinates refer to this normalized image.
- One normalized [x,y] in [0,1] becomes integer pixel coordinates in [0,w-1] and
  [0,h-1]. Reject booleans, non-finite values, extra outputs, invalid JSON/literals.
- Preserve original normalized screenshot; overlay a visible target marker and
  show a magnified crop. Export marked PNG and JSON with image hash, instruction,
  raw prediction, model revision, elapsed time, and normalized/pixel location.
- Lazy single model instance, serialized inference; loopback by default. No browser
  clicking, credentials, paid services, or automatic publishing of user uploads.
- CPU inference uses a bounded image budget and 32 generated tokens. Raw coordinate
  validity is not target correctness. ShowUI may suggest a location for absent or
  ambiguous targets; no confidence percentage or reliable abstention is promised.
- Session files get separate unpredictable directories; keep at most 32 report
  directories, expire after one hour. Gradio cache has the same age limit. Only
  immutable public examples bypass expiration.

## Evaluation and output images

Use original fictional browser interfaces: workspace settings, project board, and
file browser, with 2 English targets each. Save browser-rendered input screenshots
and manually declared target boxes independently of inference. Run six fixed tasks
once with the pinned model, record all results and timings, and score point-in-box.
Model receives only screenshot pixels and instruction, never target annotations.
No held-out/general-accuracy claim. Include failures as well as successes. No private
account screenshots in the public repository. Preserve full-resolution PNGs locally;
publish images through available authorized GitHub capabilities.

## Visual direction

Research: ShowUI official demo's input/output structure; local EvidenceDoc's
inspectable-output pattern; bundled typography/craft guidelines. Live design-reference
connector returned NO_SUBSCRIPTION; no upgrade purchased.

Reference lock: light neutral workbench, system sans, 1px borders, blue primary action,
red target marker reserved for image annotation, compact left-side instruction and
large right-side visual result. No landing-page hero or decorative imagery.

| Decision | Source | Reason |
| --- | --- | --- |
| Screenshot + instruction -> image + coordinates | ShowUI demo | Directly matches grounding task |
| Local workbench; English UI | User request and prior release preference | Test before publishing |
| One font; readable secondary copy; visible focus | Bundled typography/craft | Useful on desktop and mobile |
| Separate input and marked output; JSON evidence | EvidenceDoc pattern | Inspect model behavior |

## Success

Tests protect image validation, coordinate boundaries, image marking, report
isolation, and error behavior. At least one real model inference is inspected end
to end. README reports actual measurements and includes corresponding output
images. Local stage commits remain available even if remote write access is limited.
