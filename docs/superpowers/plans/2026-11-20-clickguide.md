# ClickGuide Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan
> task by task. Track checks and rulings in the execution ledger.

**Goal:** Locally tested screenshot grounding with stage commits and public output images.

**Architecture:** A small pure image/coordinate core feeds a lazy ShowUI adapter.
A service creates isolated reports; Gradio exposes a local workbench. Evaluation
uses original fictional screenshots and independently specified target boxes.

**Tech Stack:** Python 3.12, PyTorch CPU, Transformers 4.57.6, Pillow, Gradio 6.11.0.

**Spec:** `docs/superpowers/specs/2026-11-20-clickguide-design.md`

## Global Constraints

- English public UI/docs; no assistant attribution; existing projects untouched.
- Pinned ShowUI revision; no remote model code; local loopback by default.
- Image <=10 MiB, <=12M pixels, <=4096 per edge; instruction 3–300 characters.
- No automated clicking; suggestions are unverified until evaluated.
- Six original development tasks; report every measured outcome honestly.

## Review Focus

- Edge point 1.0 maps inside the final pixel; tested in Task 1.
- EXIF orientation, transparency, corrupt/animated images; tested in Task 1.
- Malicious or malformed coordinate strings; tested in Task 1.
- Concurrent/repeated calls and rejected predictions leave no stale result; Task 2.
- Reports expire without deleting active sessions or immutable samples; Task 2.

### Task 1: Safe image and coordinate core

Files: `clickguide/core.py`, `tests/test_core.py`, requirements and project config.
Interfaces: `load_image(path)->Image`, `parse_point(raw)->tuple[float,float]`,
`pixel_point(point,size)->tuple[int,int]`, `annotate(image,point)->Image`,
`target_crop(image,point)->Image`, `validate_instruction(text)->str`.

- [ ] Write parametrized failing tests for boundaries, corrupt input, EXIF/alpha,
  point parsing, marker placement, unchanged source, and query validation.
- [ ] Run `pytest tests/test_core.py -q`; expect missing core behavior.
- [ ] Implement minimal core, run full suite and Ruff; expect green.
- [ ] Commit `feat: validate screenshots and render target coordinates`.

### Task 2: Actual model, reports, and evaluation fixtures

Files: `clickguide/model.py`, `clickguide/service.py`, `tests/test_service.py`,
`scripts/evaluate.py`, `examples/fixtures.html`, `examples/tasks.json`.
Interfaces: `ShowUIEngine.predict(image,instruction)->str`;
`GuideService.locate(path,instruction)->dict` containing report/image/crop paths;
`score_point(point,box)->bool` uses inclusive target bounds.

- [ ] Write failing service tests with a small injected inference callable for
  successful/malformed/error responses, report metadata, pruning, and scoring.
- [ ] Run tests; confirm failures before implementation.
- [ ] Implement pinned model adapter, safe report service, and evaluation runner.
- [ ] Capture original browser screenshots; download model; run six measured tasks.
- [ ] Run full tests/Ruff and commit `feat: add pinned local ShowUI inference and evaluation`.

### Task 3: Local UI and reproducible documentation

Files: `app.py`, `tests/test_app.py`, `README.md`, `docs/EVALUATION.md`,
`docs/images/`, `.github/workflows/test.yml`, `LICENSE`, `CHANGELOG.md`.

- [ ] Write failing callback tests proving validation and model failures clear all
  previous outputs while a valid result returns annotated image and JSON.
- [ ] Implement English workbench with public examples, one-request queue and
  clear suggestion status; run full suite/Ruff and start locally.
- [ ] Inspect a real browser flow, mobile layout and output images; record measured
  limitations, include output illustrations in README and commit UI/docs stages.
- [ ] Fresh whole-branch review; fix important issues with regression tests.
- [ ] Publish only after local verification; verify remote files and CI. Preserve
  local commit history and provide it if GitHub cannot accept direct git pushes.
