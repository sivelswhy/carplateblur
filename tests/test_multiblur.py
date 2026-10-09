"""Tests for multiblur.py's model-free logic: geometry, tracking, masking and file naming."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import multiblur as mb  # noqa: E402


def options(**overrides):
    defaults = dict(mode="blur", mosaic_size=20, strength=51, boxes=False, draw_scores=False)
    return argparse.Namespace(**{**defaults, **overrides})


def textured(h=200, w=200):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# Geometry ------------------------------------------------------------------- #

def test_mode_aliases():
    assert mb.mode_name("pixelate") == "mosaic"
    assert mb.mode_name("black") == "solid"
    with pytest.raises(argparse.ArgumentTypeError):
        mb.mode_name("sparkles")


def test_enlarge_matches_deface_mask_scale():
    assert mb.enlarge((10, 10, 20, 30), 1.3) == pytest.approx((7, 4, 23, 36))


def test_no_tiles_for_images_barely_larger_than_a_tile():
    assert mb.tiles(1500, 1000, 1280) == []


def test_tiles_cover_the_whole_image_with_overlap():
    width, height, size = 4000, 3000, 1280
    covered = np.zeros((height, width), bool)
    for x, y, w, h in mb.tiles(width, height, size):
        assert w == size and h == size
        covered[y:y + h, x:x + w] = True
    assert covered.all()


def test_suppress_duplicates_keeps_best_and_drops_tile_cut_parts():
    full = ((100, 100, 200, 150), 0.9)
    shifted = ((105, 100, 205, 150), 0.6)  # same object seen by another pass
    cut = ((150, 100, 200, 150), 0.7)      # partial box at a tile edge
    other = ((400, 400, 450, 430), 0.5)
    assert mb.suppress_duplicates([shifted, cut, full, other]) == [full, other]


# Tracking ------------------------------------------------------------------- #

def face(x, y, size=40, score=0.9):
    return mb.Detection((x, y, x + size, y + size), score, True)


def test_tracker_keeps_one_mask_per_moving_object():
    tracker = mb.Tracker(max_missed=3)
    for x in range(0, 100, 10):
        masks = tracker.update([face(x, 50)])
        assert len(masks) == 1


def test_tracker_follows_the_object_while_the_detector_misses_it():
    tracker = mb.Tracker(max_missed=3)
    for x in (0, 10, 20, 30):
        tracker.update([face(x, 50)])
    predicted = tracker.update([])
    assert len(predicted) == 1
    center_x = (predicted[0].box[0] + predicted[0].box[2]) / 2
    assert 45 < center_x < 65  # moved on from x=30 (center 50) along the observed motion


def test_tracker_forgets_objects_after_max_missed_frames():
    tracker = mb.Tracker(max_missed=2)
    tracker.update([face(0, 0)])
    assert len(tracker.update([])) == 1
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []


def test_tracker_does_not_mix_plates_and_faces():
    tracker = mb.Tracker(max_missed=3)
    tracker.update([face(0, 0)])
    plate = mb.Detection((0, 0, 40, 40), 0.9, False)
    assert len(tracker.update([plate])) == 2  # the face is still predicted, the plate starts its own track


# Masking -------------------------------------------------------------------- #

def test_face_ellipse_leaves_the_corners_untouched():
    frame = textured()
    original = frame.copy()
    mb.anonymize(frame, [mb.Detection((50, 50, 150, 150), 0.9, True)], options(mode="solid"))
    assert (frame[52, 52] == original[52, 52]).all()  # corner of the box, outside the ellipse
    assert (frame[100, 100] == 0).all()               # center


def test_boxes_option_masks_the_whole_face_box():
    frame = textured()
    mb.anonymize(frame, [mb.Detection((50, 50, 150, 150), 0.9, True)], options(mode="solid", boxes=True))
    assert (frame[50:150, 50:150] == 0).all()


def test_plates_are_always_boxes():
    frame = textured()
    mb.anonymize(frame, [mb.Detection((50, 50, 150, 150), 0.9, False)], options(mode="solid"))
    assert (frame[50:150, 50:150] == 0).all()


def test_none_mode_changes_nothing():
    frame = textured()
    original = frame.copy()
    mb.anonymize(frame, [mb.Detection((50, 50, 150, 150), 0.9, True)], options(mode="none"))
    assert (frame == original).all()


def test_mosaic_uses_blocks_of_the_requested_size():
    frame = textured()
    mb.anonymize(frame, [mb.Detection((0, 0, 200, 200), 0.9, False)], options(mode="mosaic", mosaic_size=20))
    block = frame[0:20, 0:20]
    assert (block == block[0, 0]).all()


def test_boxes_outside_the_frame_are_clipped():
    frame = textured()
    mb.anonymize(frame, [mb.Detection((-50, -50, 60, 60), 0.9, False)], options(mode="solid"))
    assert (frame[0:60, 0:60] == 0).all()


def test_transparent_replacement_shows_blur_not_the_original():
    frame = textured()
    original = frame.copy()
    transparent = np.zeros((10, 10, 4), np.uint8)
    mb.anonymize(frame, [mb.Detection((0, 0, 200, 200), 0.9, False)], options(mode="image"), transparent)
    assert np.abs(frame.astype(int) - original.astype(int)).mean() > 30


# Files ---------------------------------------------------------------------- #

def test_unique_path_never_overwrites(tmp_path):
    target = tmp_path / "photo_anonymized.jpg"
    assert mb.unique_path(target) == target
    target.touch()
    assert mb.unique_path(target) == tmp_path / "photo_anonymized (1).jpg"
    (tmp_path / "photo_anonymized (1).jpg").touch()
    assert mb.unique_path(target) == tmp_path / "photo_anonymized (2).jpg"
