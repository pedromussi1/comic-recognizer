<h1 align="center">Comic Recognizer</h1>

<p align="center">
  <a href="https://www.youtube.com/watch?v=h8sp7vFeV7c"><img src="https://i.imgur.com/uPkoNw1.gif" alt="YouTube Demonstration" width="800"></a>
</p>

<p align="center">Identify a comic / graphic novel from a photo of its cover, using <b>CLIP image embeddings + FAISS</b> nearest-neighbor retrieval.</p>

<p align="center"><b>🚀 Live demo:</b> <a href="https://huggingface.co/spaces/Zao0531/comic-recognizer">huggingface.co/spaces/Zao0531/comic-recognizer</a></p>

## Why a rewrite

The original version was a MobileNetV2 classifier whose pieces never fit together — training
saved `comic_model.h5`, the app loaded `comic_book_classifier_model.h5` with 17 hardcoded
labels, and `predict.py` expected a `label_encoder.pkl` nothing produced. No dataset, model,
or encoder was committed, so it could not run.

This rewrite replaces the classifier with **retrieval**, which fits the problem much better:

| Classifier (old) | Retrieval (now) |
|---|---|
| Fixed 17 classes; adding a comic means retraining | Add a comic by embedding one cover — **no training** |
| Needs a labeled dataset + training run to work | Runs immediately from a prebuilt index |
| Broken, unrunnable wiring | End-to-end and tested |

## How it works

```
cover photo ──▶ CLIP embedding ──▶ FAISS nearest-neighbor ──▶ best-matching comic
```

- Each reference cover is embedded with **CLIP** (`clip-ViT-B-32`) into a 512-d vector.
- A **FAISS** inner-product index over L2-normalized vectors gives cosine-similarity search.
- The reference set is built from free **Open Library** cover thumbnails, with **multiple
  editions per title** so a query matches if it resembles *any* edition.

The demo index ships with ~45 well-known titles (built from Open Library).

## Real-world accuracy (and its honest limits)

Evaluated on **56 real phone photos** of an actual collection (comics on a carpet, at
angles, with glare):

| Setup | Top-1 |
|---|---|
| Open Library demo index, over titles it covers | **56%** (22/39) |
| **Enrollment** — index your *own* photos, identify held-out photos | **75%** (0.86 mean score) |

The gap is the point: Open Library often has a *different edition's* cover than the one you
own, so cross-edition matches score lower (0.65–0.74). Indexing **your own** cover photos
matches the exact editions and scores 0.77–0.94. The enrollment number is measured on real
phone photos of a **54-title personal collection** (held-out photos, not seen during
enrollment). Titles absent from the index get pulled to the nearest look-alike — a
fundamental property of pure nearest-neighbor retrieval.

The shipped index bundles the ~45-title Open Library demo set **plus that enrolled personal
collection** (~200 reference covers), so the app recognizes both famous graphic novels and
the owner's specific editions out of the box.

## Recognize your own collection (enrollment)

To reliably recognize *your* comics, enroll your own cover photos — one clear photo per
comic — then future photos match the exact editions:

```bash
# add your covers to the existing index (keeps the demo set):
python -m comicid.enroll path/to/your/cover_photos --labels labels.csv --output data/index --append
# ...or build a fresh index from only your photos (drop --append)
```

`labels.csv` is `filename,title[,author]`; without it the filename becomes the title. The
app then recognizes your collection. No training — enrollment is just embedding each cover
once. (`build_index.py` regenerates only the Open Library demo set; enrolled covers are
added on top with `--append`.)

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt   # pulls torch — sizeable first install
python app.py                     # http://127.0.0.1:5000/
```

The committed index (`data/index/`) means it runs out of the box. To rebuild or extend it:

```bash
python -m comicid.build_index     # re-fetches covers from Open Library
```

Add your own comics by editing the `COMICS` list in `comicid/build_index.py` and rebuilding.

## Layout

```
comicid/
  embedder.py      CLIP encoder (images + text)
  index.py         FAISS cover index + metadata, save/load
  recognizer.py    high-level identify()
  build_index.py   build the reference index from Open Library covers
  enroll.py        build a personal index from your own cover photos
app.py             Flask web app
data/index/        committed FAISS index + metadata (the reference set)
tests/             pytest suite
```

## Tests

```bash
python -m pytest -q
```

## Notes

- The demo index holds ~20 well-known graphic novels; recognition works best when the cover
  you photograph resembles one of them. Point `build_index.py` at your own collection to
  expand it.
- Reference covers are displayed at runtime via their Open Library URL; only the derived
  embeddings are committed, not the cover images.
