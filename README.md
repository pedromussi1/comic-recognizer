<h1 align="center">Comic Recognizer</h1>

<p align="center">
  <a href="https://www.youtube.com/watch?v=h8sp7vFeV7c"><img src="https://i.imgur.com/uPkoNw1.gif" alt="YouTube Demonstration" width="800"></a>
</p>

<p align="center">Identify a comic / graphic novel from a photo of its cover, using <b>CLIP image embeddings + FAISS</b> nearest-neighbor retrieval.</p>

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

On simulated photos (rotation, blur, brightness, crop, JPEG recompression of held-out
covers), top-1 identification was **15/15**, all above the 0.75 confidence threshold.

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
