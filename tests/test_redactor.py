"""Redactor: blurred regions are destroyed, black mode still available."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server.image_preprocessing import load_bgr
from server.redactor import redact_pages
from server.vault import Page


def _page(tmp_path):
    img = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(img).text((20, 30), "CNP 2961012331237", fill="black", font=ImageFont.truetype("arial.ttf", 40))
    p = tmp_path / "p1.png"
    img.save(p)
    return Page(1, str(p), 900, 120)


def test_blur_mode_destroys_region_without_black_box(tmp_path):
    page = _page(tmp_path)
    box = (1, 110, 30, 400, 50)
    paths, drawn = redact_pages([page], [box], "doc_t", tmp_path / "out", padding_px=4, mode="blur")
    assert drawn == 1 and paths[0].endswith("redacted_doc_t.png")
    before = load_bgr(page.base_image_path)[26:84, 106:514].astype(int)
    after = load_bgr(paths[0])[26:84, 106:514].astype(int)
    assert np.abs(after - before).mean() > 20                 # region really changed
    assert (after.max(axis=2) < 20).mean() < 0.2              # ...but is not a solid black rectangle
    gray = after.mean(axis=2)
    assert gray.std() < before.mean(axis=2).std()             # crisp glyph edges are gone
    outside_b, outside_a = load_bgr(page.base_image_path)[:, 600:], load_bgr(paths[0])[:, 600:]
    assert np.array_equal(outside_b, outside_a)               # nothing outside the box is touched


def test_black_mode_still_available(tmp_path):
    page = _page(tmp_path)
    paths, _ = redact_pages([page], [(1, 110, 30, 400, 50)], "doc_b", tmp_path / "out", mode="black")
    assert (load_bgr(paths[0])[40:70, 150:450] == 0).all()
