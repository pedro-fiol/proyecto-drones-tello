"""
tests/test_person_gallery.py

Unit tests for interceptor/person_gallery.py. Synthetic embeddings on CPU.
No YOLO, no ReID model, no drone.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from interceptor.constants import CAMERA_HORIZONTAL_FOV_DEG, GALLERY_MIN_SIGHTINGS
from interceptor.perception import Person
from interceptor.person_gallery import (
    PersonGallery,
    bearing_of_person_deg,
    greedy_assign,
)


def _unit(seed: int) -> torch.Tensor:
    """Random L2-normalized 512-D vector. Different seeds are near-orthogonal (distance ~1)."""
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(512, generator=g), p=2, dim=0)


def _noisy(base: torch.Tensor, seed: int, scale: float = 0.4) -> torch.Tensor:
    """Same person, new view: perturbation with norm ~scale. scale 0.4 -> distance ~0.07 to base."""
    g = torch.Generator().manual_seed(seed)
    return F.normalize(base + scale * torch.randn(512, generator=g) / np.sqrt(512), p=2, dim=0)


def _frame(gallery, vectors, xs=None, confidences=None, bearings=None):
    """Add one frame of sightings. Defaults: x spread left to right, conf 0.9, bearing 0."""
    n = len(vectors)
    xs = xs if xs is not None else [100.0 * (k + 1) for k in range(n)]
    confidences = confidences if confidences is not None else [0.9] * n
    bearings = bearings if bearings is not None else [0.0] * n
    return gallery.add_frame(torch.stack(vectors), xs, confidences, bearings)


def test_noisy_helper_stays_inside_threshold():
    base = _unit(1)
    assert 1.0 - float(base @ _noisy(base, 99)) < 0.30


def test_entry_hidden_until_min_sightings():
    gallery = PersonGallery()
    a = _unit(1)
    for k in range(GALLERY_MIN_SIGHTINGS - 1):
        assert _frame(gallery, [_noisy(a, k)]) == [None]
        assert gallery.snapshot() == []
    assert _frame(gallery, [_noisy(a, 100)]) == [1]
    assert [p["id"] for p in gallery.snapshot()] == [1]


def test_same_person_many_views_gives_one_card():
    gallery = PersonGallery()
    a = _unit(1)
    for k in range(20):
        _frame(gallery, [_noisy(a, k)])
    snap = gallery.snapshot()
    assert len(snap) == 1
    assert snap[0]["sightings"] == 20


def test_numbered_in_first_seen_order():
    gallery = PersonGallery()
    a, b = _unit(1), _unit(2)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, k)])
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(b, 50 + k)])
    assert [p["id"] for p in gallery.snapshot()] == [1, 2]
    assert gallery.identify(torch.stack([_noisy(b, 90), _noisy(a, 91)])) == [2, 1]


def test_first_seen_wins_even_when_confirmed_later():
    # a is seen first, then flickers out; b is confirmed first. a must still be Person 1.
    gallery = PersonGallery()
    a, b = _unit(1), _unit(2)
    _frame(gallery, [_noisy(a, 0)])
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(b, 10 + k)])
    assert gallery.identify(torch.stack([_noisy(b, 30)])) == [1]   # only b confirmed so far
    for k in range(GALLERY_MIN_SIGHTINGS - 1):
        _frame(gallery, [_noisy(a, 20 + k)])
    assert gallery.identify(torch.stack([_noisy(a, 40), _noisy(b, 41)])) == [1, 2]


def test_same_first_frame_numbered_left_to_right():
    gallery = PersonGallery()
    left, right = _unit(1), _unit(2)
    for k in range(GALLERY_MIN_SIGHTINGS):
        # right person listed first by YOLO, but sits at larger x
        _frame(gallery, [_noisy(right, k), _noisy(left, 10 + k)], xs=[800.0, 150.0])
    assert gallery.identify(torch.stack([_noisy(left, 50), _noisy(right, 51)])) == [1, 2]


def test_two_lookalikes_in_one_frame_never_merge():
    # b is inside threshold of a, but both appear in the same frame -> must stay two entries.
    gallery = PersonGallery()
    a = _unit(1)
    b = _noisy(a, 7)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, 10 + k, 0.05), _noisy(b, 20 + k, 0.05)])
    assert len(gallery.snapshot()) == 2


def test_unconfirmed_false_positive_takes_no_number():
    gallery = PersonGallery()
    _frame(gallery, [_unit(9)])   # seen once, never confirmed
    a = _unit(1)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, k)])
    assert [p["id"] for p in gallery.snapshot()] == [1]


def test_identify_labels_confirmed_only_and_does_not_mutate():
    gallery = PersonGallery()
    a, b = _unit(1), _unit(2)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, k)])
    _frame(gallery, [b])  # unconfirmed
    version = gallery.version
    ids = gallery.identify(torch.stack([_noisy(a, 77), _noisy(b, 78), _unit(3)]))
    assert ids == [1, None, None]
    assert gallery.version == version
    assert gallery.snapshot()[0]["sightings"] == GALLERY_MIN_SIGHTINGS


def test_get_embedding_is_a_copy_and_matches_person():
    gallery = PersonGallery()
    a = _unit(1)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, k)])
    emb = gallery.get_embedding(1)
    assert emb is not None
    assert 1.0 - float(emb @ a) < 0.1
    emb.zero_()
    assert float(gallery.get_embedding(1).norm()) == pytest.approx(1.0, abs=1e-5)
    assert gallery.get_embedding(2) is None
    assert gallery.has(1) and not gallery.has(2)


def test_clear_restarts_numbering_and_bumps_version():
    gallery = PersonGallery()
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(_unit(1), k)])
    version = gallery.version
    gallery.clear()
    assert gallery.snapshot() == []
    assert gallery.version > version
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(_unit(5), k)])
    assert [p["id"] for p in gallery.snapshot()] == [1]


def test_bearing_kept_from_best_confidence_sighting():
    gallery = PersonGallery()
    a = _unit(1)
    _frame(gallery, [_noisy(a, 1)], confidences=[0.85], bearings=[10.0])
    _frame(gallery, [_noisy(a, 2)], confidences=[0.95], bearings=[120.0])
    for k in range(GALLERY_MIN_SIGHTINGS - 2):
        _frame(gallery, [_noisy(a, 3 + k)], confidences=[0.90], bearings=[200.0])
    assert gallery.snapshot()[0]["bearing_deg"] == 120.0
    assert gallery.get_bearing_deg(1) == 120.0


def test_snapshot_has_no_image_or_colour_data():
    gallery = PersonGallery()
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(_unit(1), k)])
    assert set(gallery.snapshot()[0]) == {"id", "bearing_deg", "sightings", "confidence"}


def test_greedy_assign_closest_pair_wins_and_threshold_respected():
    d = torch.tensor([[0.10, 0.20],
                      [0.05, 0.50],
                      [0.90, 0.95]])
    assert greedy_assign(d, 0.30) == [1, 0, None]
    assert greedy_assign(torch.zeros((0, 2)), 0.3) == []
    assert greedy_assign(torch.zeros((2, 0)), 0.3) == [None, None]


def test_bearing_of_person_centre_and_edges():
    centre = Person(x1=430, y1=0, x2=530, y2=100, bbox_confidence=0.9)       # centre x = 480
    right_edge = Person(x1=910, y1=0, x2=1010, y2=100, bbox_confidence=0.9)  # centre x = 960
    left_edge = Person(x1=-50, y1=0, x2=50, y2=100, bbox_confidence=0.9)     # centre x = 0
    assert bearing_of_person_deg(centre, 90.0, 960) == pytest.approx(90.0)
    assert bearing_of_person_deg(right_edge, 0.0, 960) == pytest.approx(CAMERA_HORIZONTAL_FOV_DEG / 2)
    assert bearing_of_person_deg(left_edge, 0.0, 960) == pytest.approx(360.0 - CAMERA_HORIZONTAL_FOV_DEG / 2)


# ---- Demo-only: random target pick ----

import random

from interceptor.person_gallery import choose_random_person


def test_choose_random_person_never_returns_excluded():
    rng = random.Random(0)
    picks = {choose_random_person([1, 2, 3, 4], {2, 4}, rng) for _ in range(200)}
    assert picks == {1, 3}


def test_choose_random_person_covers_all_candidates():
    rng = random.Random(1)
    picks = {choose_random_person([1, 2, 3], set(), rng) for _ in range(300)}
    assert picks == {1, 2, 3}


def test_choose_random_person_none_when_exhausted():
    assert choose_random_person([1, 2], {1, 2}) is None
    assert choose_random_person([], set()) is None


def test_confirmed_ids_lists_only_confirmed_in_order():
    gallery = PersonGallery()
    a, b = _unit(1), _unit(2)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(a, k), _noisy(b, 30 + k)])
    _frame(gallery, [_unit(9)])  # unconfirmed
    assert gallery.confirmed_ids() == [1, 2]


def test_merge_duplicates_joins_split_person_and_renumbers():
    # One person, two views ~0.26 apart (e.g. half body vs full body). Scanned with a tight threshold
    # they split into two entries; merge with the normal 0.30 joins them back.
    a, b = _unit(1), _unit(2)
    view_1 = a
    view_2 = _noisy(a, 7, scale=0.9)
    assert 0.20 < 1.0 - float(view_1 @ view_2) < 0.30
    gallery = PersonGallery(match_threshold=0.15)
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(view_1, k, 0.05)], xs=[100.0])
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(b, 40 + k, 0.05)], xs=[100.0])
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(view_2, 80 + k, 0.05)], xs=[100.0])
    assert len(gallery.snapshot()) == 3
    version = gallery.version
    gallery.match_threshold = 0.30
    assert gallery.merge_duplicates() == 1
    snap = gallery.snapshot()
    assert [p["id"] for p in snap] == [1, 2]
    assert snap[0]["sightings"] == 2 * GALLERY_MIN_SIGHTINGS
    assert gallery.version > version
    assert gallery.identify(torch.stack([_noisy(view_2, 99, 0.05), _noisy(b, 98, 0.05)])) == [1, 2]


def test_merge_duplicates_keeps_different_people():
    gallery = PersonGallery()
    for k in range(GALLERY_MIN_SIGHTINGS):
        _frame(gallery, [_noisy(_unit(1), k), _noisy(_unit(2), 50 + k)])
    assert gallery.merge_duplicates() == 0
    assert len(gallery.snapshot()) == 2
