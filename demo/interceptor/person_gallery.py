"""Scan gallery: one entry per distinct person seen during the takeoff 360° scan.

Privacy rule: an entry never holds an image, crop or colour, not in RAM and not on disk.
It holds only:
    - a ReID embedding (512 floats, mean of all sightings)
    - bearing in degrees, clockwise from the heading where the scan started (search spin direction)

Numbering: confirmed entries are Person 1, 2, ... in first-seen order. Persons first seen in the
same frame are numbered left to right. A CW scan therefore numbers the room left to right.

Public:
    bearing_of_person_deg(person, heading_deg, frame_width_px) -> float in [0, 360)
    greedy_assign(distances, threshold) -> list[Optional[int]]
    choose_random_person(candidate_ids, exclude_ids, rng) -> Optional[int]
    PersonGallery: add_frame, merge_duplicates, identify, get_embedding, get_bearing_deg, has, confirmed_ids,
                   snapshot, clear, version
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from interceptor.constants import (
    CAMERA_HORIZONTAL_FOV_DEG,
    GALLERY_MATCH_THRESHOLD,
    GALLERY_MIN_SIGHTINGS,
)

if TYPE_CHECKING:
    import torch
    from interceptor.perception import Person


def bearing_of_person_deg(person: "Person", heading_deg: float, frame_width_px: int) -> float:
    """Bearing of the person in [0, 360): drone heading + bbox-centre offset across the camera FOV.
    heading_deg is clockwise from the scan start heading.
    """
    offset_ratio = (person.bbox_center_x - frame_width_px / 2) / frame_width_px
    return (heading_deg + offset_ratio * CAMERA_HORIZONTAL_FOV_DEG) % 360.0


def greedy_assign(distances: "torch.Tensor", threshold: float) -> list[Optional[int]]:
    """Unique row -> column assignment, closest pairs first.
    distances: (rows, cols) cosine distances. A pair above threshold is never assigned.
    Each column gets at most one row, so two people in one frame never merge into one entry.
    Returns, per row, the assigned column index or None.
    """
    rows, cols = distances.shape
    result: list[Optional[int]] = [None] * rows
    if rows == 0 or cols == 0:
        return result

    # One device->host copy, not one GPU sync per element.
    distance_rows = distances.detach().cpu().tolist()
    pairs = []
    for r in range(rows):
        for c in range(cols):
            d = distance_rows[r][c]
            if d <= threshold:
                pairs.append((d, r, c))
    pairs.sort()

    used_cols: set[int] = set()
    for _, r, c in pairs:
        if result[r] is None and c not in used_cols:
            result[r] = c
            used_cols.add(c)
    return result


def choose_random_person(candidate_ids: list[int], exclude_ids: set[int],
                         rng: random.Random = random) -> Optional[int]:
    """Uniform random pick from candidate_ids minus exclude_ids. None when nothing is left."""
    pool = [pid for pid in candidate_ids if pid not in exclude_ids]
    if not pool:
        return None
    return rng.choice(pool)


@dataclass
class GalleryEntry:
    """One person in the gallery. person_id stays None until the entry has GALLERY_MIN_SIGHTINGS."""

    embedding_sum: "torch.Tensor"   # sum of L2-normalized sightings; mean direction = normalize(sum)
    first_seen_seq: int             # creation order; sets the Person number
    bearing_deg: float              # bearing at the best-confidence sighting
    best_confidence: float
    sightings: int
    person_id: Optional[int] = None

    @property
    def embedding(self) -> "torch.Tensor":
        """Mean embedding, L2-normalized."""
        import torch.nn.functional as F
        return F.normalize(self.embedding_sum, p=2, dim=0)


class PersonGallery:
    """Thread-safe store of scanned persons. The video loop writes, the webapp thread reads.
    version increases on every change the webapp can see (new card, clear).
    """

    def __init__(self, match_threshold: float = GALLERY_MATCH_THRESHOLD,
                 min_sightings: int = GALLERY_MIN_SIGHTINGS) -> None:
        self.match_threshold = match_threshold
        self.min_sightings = min_sightings
        self._entries: list[GalleryEntry] = []
        self._next_seq = 0
        self._lock = threading.Lock()
        self.version = 0

    def clear(self) -> None:
        """Drop every entry. Numbering restarts at Person 1."""
        with self._lock:
            self._entries = []
            self._next_seq = 0
            self.version += 1

    def _renumber(self) -> None:
        """Confirmed entries -> 1..n by first-seen order. Caller holds _lock.
        A late-confirmed entry that was seen earlier shifts later numbers by one (scan phase only).
        """
        confirmed = sorted((e for e in self._entries if e.sightings >= self.min_sightings),
                           key=lambda e: e.first_seen_seq)
        for number, entry in enumerate(confirmed, start=1):
            entry.person_id = number

    def add_frame(self, embeddings: "torch.Tensor",
                  center_x_px: list[float],
                  confidences: list[float],
                  bearings_deg: list[float]) -> list[Optional[int]]:
        """Add one frame of sightings (scan phase only).
        embeddings: (N, 512) L2-normalized. center_x_px/confidences/bearings_deg: length N.
        Each sighting joins the closest entry within match_threshold, else starts a new entry.
        New entries in one frame get first-seen order left to right (center_x_px ascending).
        Returns per-sighting person_id (None while the entry is not confirmed yet).
        """
        import torch

        n = embeddings.shape[0]
        with self._lock:
            if self._entries:
                # .to() per entry — entries may sit on another device after a cuda/cpu swap.
                gallery_matrix = torch.stack([e.embedding.to(embeddings.device) for e in self._entries])
                distances = 1.0 - embeddings @ gallery_matrix.T
                assignment = greedy_assign(distances, self.match_threshold)
            else:
                assignment = [None] * n

            entry_of_sighting: list[Optional[GalleryEntry]] = [None] * n
            for i in range(n):
                if assignment[i] is not None:
                    entry_of_sighting[i] = self._entries[assignment[i]]

            # New persons, left to right.
            for i in sorted((i for i in range(n) if assignment[i] is None), key=lambda i: center_x_px[i]):
                entry = GalleryEntry(
                    embedding_sum=torch.zeros_like(embeddings[i]),
                    first_seen_seq=self._next_seq,
                    bearing_deg=bearings_deg[i],
                    best_confidence=confidences[i],
                    sightings=0,
                )
                self._next_seq += 1
                self._entries.append(entry)
                entry_of_sighting[i] = entry

            newly_confirmed = False
            for i in range(n):
                entry = entry_of_sighting[i]
                entry.embedding_sum = entry.embedding_sum.to(embeddings.device) + embeddings[i].detach()
                if confidences[i] > entry.best_confidence:
                    entry.best_confidence = confidences[i]
                    entry.bearing_deg = bearings_deg[i]
                entry.sightings += 1
                if entry.sightings == self.min_sightings:
                    newly_confirmed = True

            if newly_confirmed:
                self._renumber()
                self.version += 1

            return [entry_of_sighting[i].person_id for i in range(n)]

    def merge_duplicates(self) -> int:
        """Merge confirmed entries whose mean embeddings match (one person split during the scan).
        Closest pair first, repeat until no pair is within match_threshold. The earlier-seen entry keeps
        its place in the numbering. Returns the number of merges.
        """
        merges = 0
        with self._lock:
            while True:
                confirmed = [e for e in self._entries if e.sightings >= self.min_sightings]
                best = None
                for i in range(len(confirmed)):
                    for j in range(i + 1, len(confirmed)):
                        a = confirmed[i].embedding
                        b = confirmed[j].embedding.to(a.device)
                        distance = 1.0 - float(a @ b)
                        if distance <= self.match_threshold and (best is None or distance < best[0]):
                            best = (distance, confirmed[i], confirmed[j])
                if best is None:
                    break
                _, first, second = best
                keep, drop = (first, second) if first.first_seen_seq <= second.first_seen_seq else (second, first)
                keep.embedding_sum = keep.embedding_sum + drop.embedding_sum.to(keep.embedding_sum.device)
                keep.sightings += drop.sightings
                if drop.best_confidence > keep.best_confidence:
                    keep.best_confidence = drop.best_confidence
                    keep.bearing_deg = drop.bearing_deg
                # By identity — dataclass == would compare the embedding tensors.
                self._entries = [e for e in self._entries if e is not drop]
                merges += 1
            if merges:
                for e in self._entries:
                    e.person_id = None
                self._renumber()
                self.version += 1
        return merges

    def identify(self, embeddings: "torch.Tensor") -> list[Optional[int]]:
        """Label each embedding with a confirmed person_id, or None. Does not change the gallery."""
        import torch

        with self._lock:
            confirmed = [e for e in self._entries if e.person_id is not None]
            if not confirmed or embeddings.shape[0] == 0:
                return [None] * embeddings.shape[0]
            gallery_matrix = torch.stack([e.embedding.to(embeddings.device) for e in confirmed])
            distances = 1.0 - embeddings @ gallery_matrix.T
            assignment = greedy_assign(distances, self.match_threshold)
            return [confirmed[c].person_id if c is not None else None for c in assignment]

    def _find(self, person_id: int) -> Optional[GalleryEntry]:
        for e in self._entries:
            if e.person_id == person_id:
                return e
        return None

    def has(self, person_id: int) -> bool:
        """True when a confirmed entry has this person_id."""
        with self._lock:
            return self._find(person_id) is not None

    def confirmed_ids(self) -> list[int]:
        """Person numbers of all confirmed entries, ascending."""
        with self._lock:
            return sorted(e.person_id for e in self._entries if e.person_id is not None)

    def get_embedding(self, person_id: int) -> Optional["torch.Tensor"]:
        """Copy of the mean embedding of a confirmed entry, or None."""
        with self._lock:
            entry = self._find(person_id)
            return entry.embedding.detach().clone() if entry is not None else None

    def get_bearing_deg(self, person_id: int) -> Optional[float]:
        """Scan bearing of a confirmed entry, or None."""
        with self._lock:
            entry = self._find(person_id)
            return entry.bearing_deg if entry is not None else None

    def snapshot(self) -> list[dict]:
        """Confirmed entries as JSON-ready dicts, sorted by person_id. No image or colour data."""
        with self._lock:
            confirmed = sorted((e for e in self._entries if e.person_id is not None),
                               key=lambda e: e.person_id)
            return [
                {
                    "id":          e.person_id,
                    "bearing_deg": round(e.bearing_deg, 1),
                    "sightings":   e.sightings,
                    "confidence":  round(e.best_confidence, 3),
                }
                for e in confirmed
            ]
