# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-07-07

### Added
- **Live hosted demo** on Hugging Face Spaces:
  https://huggingface.co/spaces/Zao0531/comic-recognizer
- `Dockerfile` for containerized hosting (gunicorn on port 7860; writable CLIP-model cache),
  and a GitHub Action that auto-syncs the app + index to the Space on every push to `main`.

## [2.2.1] - 2026-07-07

### Fixed
- De-lumped the enrolled collection: distinct volumes/runs are now separate titles (54 -> 64),
  and corrected mislabels (a Straczynski Thor and a Waid Daredevil had been filed under the
  wrong creator; Doomsday Clock Part 1/2 and Invincible Compendium One/Two/Three had been merged).

## [2.2.0] - 2026-07-07

### Added
- `enroll --append` to add your own covers to the existing index instead of replacing it.
- The shipped index now bundles the Open Library demo set **plus an enrolled 54-title
  personal collection** (~200 reference covers), so the app recognizes real-world comics
  (Batman New 52 / Rebirth, the Flash, Astonishing X-Men, etc.) out of the box.
- `data/collection_labels.csv` documents the enrolled collection.

### Changed
- Real-world evaluation updated: **75% top-1** (0.86 mean score) on held-out photos of a
  54-title personal collection.

## [2.1.0] - 2026-07-07

### Added
- **Enrollment** (`comicid/enroll.py` + `python -m comicid.enroll`): build a personal index
  from your own cover photos (one per comic), so recognition matches your exact editions.
  On real phone photos this reached ~66% top-1 (scores 0.77–0.94), vs ~56% for the
  generic Open Library index (whose different editions score 0.65–0.74).
- Expanded the demo index to ~45 titles / 121 covers (added many mainstream Marvel/DC/Image
  graphic novels).

### Changed
- README now reports an honest real-world evaluation on 56 real photos and documents the
  nearest-neighbor limitation (titles absent from the index match their nearest look-alike).

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

[2.3.0]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.3.0
[2.2.1]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.2.1
[2.2.0]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.2.0
[2.1.0]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.1.0
[2.0.0]: https://github.com/pedromussi1/comic-recognizer/releases/tag/v2.0.0
