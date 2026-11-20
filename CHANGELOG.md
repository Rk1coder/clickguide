# Changelog

## 0.1.0 — 2026-11-20

- Established a separate local-first project and staged Git history.
- Added bounded image validation, strict coordinate parsing, and non-destructive target markers.
- Measured pinned ShowUI inference on six original tasks; retained all outputs, including misses.
- Added a Qwen3-VL comparison, native coordinate normalization, and raw prediction diagnostics.
- Selected Qwen3-VL as the default after its 4/5 result on valid development tasks against ShowUI’s 3/5.
- Added an English local workbench, original-image preview, magnified target, PNG/JSON downloads, and failure-state handling.
- Added regression tests and continuous integration without model downloads.
- Archived a mislabeled sixth fixture unchanged and excluded it from current metrics.
- Added regression coverage for stale results, duplicate predictions, PNG integrity, and restart-safe private cache expiration.
