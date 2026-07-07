"""Flask app: identify a comic from a photo of its cover.

Rewritten from a broken MobileNetV2 classifier (whose train/serve wiring never matched and
which shipped no model or dataset) into a CLIP + FAISS **retrieval** system: the uploaded
cover is embedded and matched against a reference index of covers. See comicid/ for the core
and comicid/build_index.py to (re)build the index.
"""

import os
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from comicid import CoverRecognizer

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB cap

# Loads the CLIP model + FAISS index once at startup.
recognizer = CoverRecognizer()


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            return render_template("index.html", error="Please choose an image to upload.")
        if not _allowed(file.filename):
            return render_template("index.html", error="Unsupported file type.")

        ext = os.path.splitext(secure_filename(file.filename))[1].lower()
        stored_name = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(file_path)

        matches = recognizer.identify(file_path, k=3)
        best = matches[0] if matches else None
        return render_template(
            "result.html",
            image=stored_name,
            best=best,
            confident=bool(best and best.confident),
            others=matches[1:] if matches else [],
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
