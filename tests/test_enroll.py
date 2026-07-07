"""Enrollment label parsing (no CLIP)."""

from comicid.enroll import _load_labels


def test_load_labels_reads_title_and_author(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,title,author\n"
        "a.jpg,Watchmen,Alan Moore\n"
        "sub/b.jpg,Maus,Art Spiegelman\n",
        encoding="utf-8",
    )
    labels = _load_labels(str(csv_path))
    assert labels["a.jpg"]["title"] == "Watchmen"
    assert labels["a.jpg"]["author"] == "Alan Moore"
    # paths are keyed by basename so a labels file can list nested paths
    assert labels["b.jpg"]["title"] == "Maus"


def test_load_labels_tolerates_missing_author(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("filename,title\nx.png,Bone\n", encoding="utf-8")
    labels = _load_labels(str(csv_path))
    assert labels["x.png"]["title"] == "Bone"
    assert labels["x.png"]["author"] == ""
