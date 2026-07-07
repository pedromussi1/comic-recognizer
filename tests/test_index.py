"""CoverIndex: ranking correctness and save/load round-trip (no CLIP)."""

import numpy as np

from comicid.index import CoverIndex


def _unit(vec):
    v = np.asarray(vec, dtype="float32")
    return v / np.linalg.norm(v)


def _build():
    idx = CoverIndex(dim=3)
    idx.add(
        np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([1, 1, 0])]),
        [{"title": "A"}, {"title": "B"}, {"title": "C"}],
    )
    return idx


def test_query_returns_nearest_first():
    idx = _build()
    results = idx.query(_unit([1, 0, 0]), k=3)
    assert results[0][0]["title"] == "A"          # exact match on top
    assert results[0][1] == max(s for _, s in results)  # sorted by score desc
    assert abs(results[0][1] - 1.0) < 1e-5         # cosine of identical vectors


def test_query_respects_k_and_empty():
    assert len(_build().query(_unit([1, 0, 0]), k=2)) == 2
    assert CoverIndex(dim=3).query(_unit([1, 0, 0])) == []


def test_save_load_round_trip(tmp_path):
    idx = _build()
    idx.save(str(tmp_path))
    loaded = CoverIndex.load(str(tmp_path))
    assert len(loaded) == 3
    assert [m["title"] for m in loaded.metadata] == ["A", "B", "C"]
    assert loaded.query(_unit([0, 1, 0]), k=1)[0][0]["title"] == "B"
