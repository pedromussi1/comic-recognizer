# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-07

Complete rewrite: from a broken MobileNetV2 classifier into a CLIP + FAISS retrieval system.

### Added
- `comicid/` package: CLIP embedder, FAISS cover index (save/load), high-level recognizer,
  and an index builder that pulls free cover thumbnails from Open Library (multiple editions
  per title for robustness).
- Committed reference index (`data/index/`) so the app runs out of the box.
- Redesigned, responsive, dark-mode-aware web UI (drag-and-drop upload, image preview,
  match with reference cover, confidence, and alternatives).
- `pytest` suite, pinned `requirements.txt`, `.gitignore`.

### Removed
- The broken CNN pipeline (`src/model.py`, `src/predict.py`, `src/preprocess.py`,
  `src/utils.py`) whose train/serve wiring never matched and which shipped no model/dataset.

### Fixed
- The project now actually runs end-to-end. On simulated photos of held-out covers, top-1
  identification was 15/15, all above the confidence threshold.
- Replaced the bloated UTF-16 `requirements.txt` (325 unrelated packages) with a minimal one.

[2.0.0]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.0.0
