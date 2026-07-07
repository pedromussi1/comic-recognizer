"""CoverRecognizer wiring, with a stub embedder (no CLIP download)."""

import numpy as np

from comicid.index import CoverIndex
from comicid.recognizer import CoverRecognizer, Match


def test_match_confident_threshold():
    assert Match("T", "A", 0.80, "", "").confident
    assert not Match("T", "A", 0.60, "", "").confident


class _StubEmbedder:
    """Returns a fixed embedding regardless of image, so identify() needs no CLIP/image."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype="float32")

    def embed_image_path(self, path):
        return self._vec / np.linalg.norm(self._vec)


def test_identify_returns_nearest_match(tmp_path):
    idx = CoverIndex(dim=3)
    idx.add(
        np.stack([[1, 0, 0], [0, 1, 0]]).astype("float32"),
        [{"title": "Watchmen", "author": "Alan Moore", "cover_image": "u", "source_url": "s"},
         {"title": "Maus", "author": "Art Spiegelman", "cover_image": "u2", "source_url": "s2"}],
    )
    idx.save(str(tmp_path))

    rec = CoverRecognizer(index_dir=str(tmp_path), embedder=_StubEmbedder([0.9, 0.1, 0]))
    matches = rec.identify("ignored.jpg", k=2)
    assert matches[0].title == "Watchmen"
    assert matches[0].author == "Alan Moore"
    assert matches[0].score > matches[1].score
