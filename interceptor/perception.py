"""
interceptor/perception.py

YOLOv8-pose wrapper. Pure perception layer: takes a BGR camera frame, returns
structured `Person` detections. No Tello SDK calls, no threading, no drawing.

Exposed keypoints: nose, eyes, ears, shoulders. The orbit logic in Phase 7 needs
the ears to decide which side to fly around a person; exposing them now keeps
the data model stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from ultralytics import YOLO

from interceptor.constants import (
    NOSE_CONFIDENCE_THRESHOLD,
    NOSE_KEYPOINT_INDEX,
    YOLO_CONFIDENCE_MIN,
    YOLO_MODEL_PATH,
    YOLO_PERSON_CLASS_ID,
)


# COCO keypoint indices (17-point pose schema)
LEFT_EYE_KEYPOINT_INDEX = 1
RIGHT_EYE_KEYPOINT_INDEX = 2
LEFT_EAR_KEYPOINT_INDEX = 3
RIGHT_EAR_KEYPOINT_INDEX = 4
LEFT_SHOULDER_KEYPOINT_INDEX = 5
RIGHT_SHOULDER_KEYPOINT_INDEX = 6

KEYPOINT_VISIBILITY_THRESHOLD = 0.5


@dataclass
class Keypoint:
    x: int
    y: int
    confidence: float

    @property
    def is_visible(self) -> bool:
        return self.confidence >= KEYPOINT_VISIBILITY_THRESHOLD


@dataclass
class Person:
    """One detected person with bbox and the subset of keypoints we use."""

    x1: int
    y1: int
    x2: int
    y2: int
    bbox_confidence: float
    # BoT-SORT tracking ID. None if tracker disabled or detection has no track
    # this frame (just appeared / re-identified). Used for single-person identity
    # lock — orchestrator stores a locked id at enrollment and filters detections.
    track_id: Optional[int] = None
    nose: Optional[Keypoint] = None
    left_eye: Optional[Keypoint] = None
    right_eye: Optional[Keypoint] = None
    left_ear: Optional[Keypoint] = None
    right_ear: Optional[Keypoint] = None
    left_shoulder: Optional[Keypoint] = None
    right_shoulder: Optional[Keypoint] = None

    @property
    def bbox_center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def bbox_center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def bbox_width_pixels(self) -> int:
        return self.x2 - self.x1

    @property
    def bbox_height_pixels(self) -> int:
        return self.y2 - self.y1

    @property
    def bbox_area_pixels(self) -> int:
        return self.bbox_width_pixels * self.bbox_height_pixels

    @property
    def face_visible(self) -> bool:
        """Strict gate: face counts as visible only when nose passes NOSE_CONFIDENCE_THRESHOLD."""
        return self.nose is not None and self.nose.confidence >= NOSE_CONFIDENCE_THRESHOLD

    @property
    def target_x_pixels(self) -> int:
        """X target pixel for control: nose if face is visible, else bbox center."""
        if self.face_visible and self.nose is not None:
            return self.nose.x
        return self.bbox_center_x

    @property
    def target_y_pixels(self) -> int:
        """Y target pixel for control: nose if face is visible, else bbox center."""
        if self.face_visible and self.nose is not None:
            return self.nose.y
        return self.bbox_center_y


class PersonDetector:
    """
    Stateless YOLOv8-pose runner.

    Caller owns frame striding (`YOLO_FRAME_STRIDE`), camera capture and threading.
    This class only knows how to turn a frame into `Person` detections.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_persons_in_frame(self, frame) -> list[Person]:
        """Run BoT-SORT tracking on one BGR frame and return tracked persons.

        Uses model.track(persist=True) so IDs survive across frames. IDs only
        valid for the lifetime of this PersonDetector instance — recreating the
        detector resets the tracker counter. Detections without an ID this frame
        (just-appeared) get track_id=None.
        """
        results = self.model.track(
            frame,
            classes=[YOLO_PERSON_CLASS_ID],
            conf=YOLO_CONFIDENCE_MIN,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
        )
        persons: list[Person] = []
        for result in results:
            persons.extend(self._build_persons_from_result(result))
        return persons

    def _build_persons_from_result(self, result) -> list[Person]:
        persons: list[Person] = []
        if result.boxes is None:
            return persons
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id: Optional[int] = None
            if box.id is not None:
                try:
                    track_id = int(box.id[0])
                except (IndexError, TypeError):
                    track_id = None
            person = Person(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                bbox_confidence=float(box.conf[0]),
                track_id=track_id,
            )
            self._attach_keypoints_to_person(person, result, index)
            persons.append(person)
        return persons

    def _attach_keypoints_to_person(self, person: Person, result, person_index: int) -> None:
        if result.keypoints is None or person_index >= len(result.keypoints):
            return
        keypoints = result.keypoints[person_index]
        person.nose = self._extract_keypoint_at_index(keypoints, NOSE_KEYPOINT_INDEX)
        person.left_eye = self._extract_keypoint_at_index(keypoints, LEFT_EYE_KEYPOINT_INDEX)
        person.right_eye = self._extract_keypoint_at_index(keypoints, RIGHT_EYE_KEYPOINT_INDEX)
        person.left_ear = self._extract_keypoint_at_index(keypoints, LEFT_EAR_KEYPOINT_INDEX)
        person.right_ear = self._extract_keypoint_at_index(keypoints, RIGHT_EAR_KEYPOINT_INDEX)
        person.left_shoulder = self._extract_keypoint_at_index(
            keypoints, LEFT_SHOULDER_KEYPOINT_INDEX
        )
        person.right_shoulder = self._extract_keypoint_at_index(
            keypoints, RIGHT_SHOULDER_KEYPOINT_INDEX
        )

    @staticmethod
    def _extract_keypoint_at_index(keypoints, index: int) -> Optional[Keypoint]:
        try:
            confidence = float(keypoints.conf[0][index])
            x = int(keypoints.xy[0][index][0])
            y = int(keypoints.xy[0][index][1])
        except (IndexError, TypeError):
            return None
        return Keypoint(x=x, y=y, confidence=confidence)
