# Code Breakdown

Comic Recognizer identifies a comic from a photo of its cover using **image retrieval**:
a query cover is embedded with CLIP and matched against a reference index of covers with
FAISS nearest-neighbor search. There is **no trained model and no classes** — adding a comic
just means embedding one more cover.

> This replaced the project's original (broken) MobileNetV2 classifier. See `CHANGELOG.md`.

## Pipeline

```
cover photo ──▶ CLIP embedding ──▶ FAISS nearest-neighbor ──▶ best-matching comic
```

Everything lives in the `comicid/` package; `app.py` is a thin Flask layer on top.

## `comicid/embedder.py` — CLIP encoder
Wraps `sentence-transformers`' `clip-ViT-B-32`. `embed_images()` returns L2-normalized
512-d vectors (so an inner-product search equals cosine similarity). The model loads lazily
on first use, so importing the package is cheap and the web app starts fast.

## `comicid/index.py` — the FAISS cover index
`CoverIndex` holds a `faiss.IndexFlatIP` plus a parallel list of per-cover metadata (title,
author, reference-cover URL, source link). Key methods:
- `add(embeddings, metadata)` — insert covers.
- `query(embedding, k)` — return the top-k `(metadata, cosine_similarity)` pairs.
- `save(dir)` / `load(dir)` — persist to `cover.index` + `metadata.json`.

Exact search is instant at this scale (a few thousand covers); the same code swaps to an
approximate index (IVF/HNSW) if the reference set ever grows huge.

## `comicid/recognizer.py` — high-level API
`CoverRecognizer.identify(image_path, k)` embeds the image, queries the index, and returns a
list of `Match` objects. A `Match.confident` flag (cosine ≥ 0.75) drives the web app's
"Match" vs "closest guess" label.

## Building the reference index
Three complementary sources feed the shipped index (~3,700 covers):
- **`comicid/build_index.py`** — fetches free cover thumbnails from **Open Library** for a
  curated list of well-known graphic novels (multiple editions per title for robustness).
- **`comicid/enroll.py`** — builds a **personal** index from your own cover photos
  (`python -m comicid.enroll <folder> --labels labels.csv`), matching your exact editions.
- **`comicid/ingest_metron.py`** — pulls thousands of real **issue** covers from the
  **Metron** API (targets popular series; saves after each series so it is resumable; backs
  off on rate limits). This is what scales the recognizer toward general "any cover" use.

## `app.py` — the web app
A small Flask app: it validates the upload, saves it, calls `recognizer.identify()`, and
renders the best match (with the reference cover, similarity score, and alternatives) using
the templates in `templates/`. The recognizer (and its index) is created once at startup.

## Tests
`tests/` covers the index ranking + save/load, the recognizer wiring (with a stub embedder,
so no model download), and the enrollment label parsing — all without network or GPU.
