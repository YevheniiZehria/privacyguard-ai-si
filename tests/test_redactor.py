"""Redactor: blurred regions are destroyed at every strength, black mode still available, parameters adapt."""
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from server.image_preprocessing import load_bgr
from server.redactor import (RESIDUAL_HP_THRESHOLD, Redactor, blur_region, effective_padding, profile,
                             redact_pages, residual_detail, resolve_strength)
from server.vault import Page

VALUE = "2961012331237"


def _page(tmp_path):
    img = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(img).text((20, 30), f"CNP {VALUE}", fill="black", font=ImageFont.truetype("arial.ttf", 40))
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
    paths, _ = redact_pages([page], [(1, 110, 30, 400, 50)], "doc_b2", tmp_path / "out", mode="black",
                            blur_strength="low")             # strength is irrelevant in black mode
    assert (load_bgr(paths[0])[40:70, 150:450] == 0).all()


def test_padding_scales_with_box_and_is_capped():
    assert effective_padding(40, 4, 1500) == 10                # 25% of a 40 px text line
    assert effective_padding(8, 4, 1500) >= 4                  # never below the minimum
    assert effective_padding(1000, 4, 600) == 9                # big box on a small scan: capped at 1.5% of 600
    assert effective_padding(40, 4, 1500, scale=1.5) == 15     # padding multiplier
    assert Redactor(blur_strength="maximum").padding_scale == 1.5
    assert Redactor(blur_strength="high").padding_scale == 1.0


@pytest.mark.parametrize("value,level", [("low", 3), ("medium", 5), ("high", 7), ("maximum", 10), ("MAX", 10),
                                         (1, 1), (10, 10), ("4", 4), (None, 7)])
def test_resolve_strength(value, level):
    assert resolve_strength(value) == level


@pytest.mark.parametrize("bad", [0, 11, "ultra", 2.5, True, ""])
def test_resolve_strength_rejects(bad):
    with pytest.raises(ValueError):
        resolve_strength(bad)


def test_profile_grows_with_level_resolution_and_text_height():
    roi = np.full((60, 400, 3), 255, np.uint8)
    roi[20:40, 20:380] = 0                                    # high-contrast "text"
    profs = [profile(lv, roi, 40, 1600) for lv in range(1, 11)]
    for a, b in zip(profs, profs[1:]):
        assert b.block >= a.block and b.kernel >= a.kernel and b.noise > a.noise
    assert profs[0].block >= 0.35 * 40                        # safety floor even at level 1
    assert profile(7, roi, 10, 4000).block > profile(7, roi, 10, 1000).block   # resolution-aware
    assert profile(7, roi, 160, 1600).block > profile(7, roi, 40, 1600).block   # font-size-aware
    faint = np.full((60, 400, 3), 230, np.uint8)
    assert profile(7, roi, 40, 1600).noise > profile(7, faint, 40, 1600).noise  # contrast-aware


def _render(width: int, font_px: int):
    font = ImageFont.truetype("arial.ttf", font_px)
    img = Image.new("RGB", (width, font_px * 3), "white")
    d = ImageDraw.Draw(img)
    x0, y0 = 40, font_px
    d.text((x0, y0), VALUE, fill="black", font=font)
    l, t, r, b = d.textbbox((x0, y0), VALUE, font=font)
    return np.array(img)[:, :, ::-1].copy(), (l, t, r - l, b - t)


@pytest.mark.parametrize("width,font_px", [(1600, 40), (1600, 160), (4000, 40), (4000, 160)])
def test_every_level_is_unreadable(width, font_px):
    """Small / large fonts on scan-sized and 4K-sized canvases: no level leaves a readable digit run."""
    from server.ocr_engine import ocr_image
    base, (x, y, w, h) = _render(width, font_px)
    rng = np.random.default_rng(0)
    for level in range(1, 11):
        img = base.copy()
        pad = Redactor(blur_strength=level).padding(h, max(img.shape[:2]))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), x + w + pad + 1, y + h + pad + 1
        blur_region(img, x0, y0, x1, y1, rng, level, text_h=h)
        assert residual_detail(img[y0:y1, x0:x1], h) <= RESIDUAL_HP_THRESHOLD, level
        _, words = ocr_image(img, deskew=False)
        text = re.sub(r"[^0-9]", "", " ".join(wd.text for wd in words))
        runs = [VALUE[i:i + 4] for i in range(len(VALUE) - 3)]
        assert not any(r in text for r in runs), f"level {level}: readable digits {text!r}"
