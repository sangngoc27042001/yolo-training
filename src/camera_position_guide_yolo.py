"""
Camera Position Guide for AI Proctoring (YOLO Version)

Uses YOLO pose estimation model (TFLite) instead of MediaPipe.

Guides candidates to properly position their phone camera at a 45° side angle
to capture upper body and desk workspace.

Requirements:
- Phone placed on either side of candidate
- 45° angle view (diagonal showing profile and desk)
- Upper body visible (face, shoulders, hands)
- Desk/workspace visible
"""

import cv2
import numpy as np
import math
import time
import threading
import tensorflow as tf
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path


# ── YOLO Pose Keypoint indices (COCO 17-keypoint format) ──
class PoseKeypoint:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Skeleton connections for drawing
SKELETON = [
    (PoseKeypoint.NOSE, PoseKeypoint.LEFT_EYE),
    (PoseKeypoint.NOSE, PoseKeypoint.RIGHT_EYE),
    (PoseKeypoint.LEFT_EYE, PoseKeypoint.LEFT_EAR),
    (PoseKeypoint.RIGHT_EYE, PoseKeypoint.RIGHT_EAR),
    (PoseKeypoint.NOSE, PoseKeypoint.LEFT_SHOULDER),
    (PoseKeypoint.NOSE, PoseKeypoint.RIGHT_SHOULDER),
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER),
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.LEFT_ELBOW),
    (PoseKeypoint.LEFT_ELBOW, PoseKeypoint.LEFT_WRIST),
    (PoseKeypoint.RIGHT_SHOULDER, PoseKeypoint.RIGHT_ELBOW),
    (PoseKeypoint.RIGHT_ELBOW, PoseKeypoint.RIGHT_WRIST),
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.LEFT_HIP),
    (PoseKeypoint.RIGHT_SHOULDER, PoseKeypoint.RIGHT_HIP),
    (PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP),
    (PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_KNEE),
    (PoseKeypoint.LEFT_KNEE, PoseKeypoint.LEFT_ANKLE),
    (PoseKeypoint.RIGHT_HIP, PoseKeypoint.RIGHT_KNEE),
    (PoseKeypoint.RIGHT_KNEE, PoseKeypoint.RIGHT_ANKLE),
]

SKELETON_COLOR = (0, 255, 0)
KEYPOINT_COLOR = (0, 0, 255)
KEYPOINT_CONF_THRESHOLD = 0.5


# ── YOLO Pose TFLite Inference ──

class YoloPoseEstimator:
    """Runs YOLO pose estimation via TFLite."""

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]['shape']
        self.input_size = (input_shape[2], input_shape[1])  # (width, height)

    def _preprocess(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess a BGR OpenCV frame for the model."""
        orig_h, orig_w = frame_rgb.shape[:2]
        tw, th = self.input_size

        scale = min(tw / orig_w, th / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_x = (tw - new_w) // 2
        pad_y = (th - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, pad_x, pad_y

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int,
                     scale: float, pad_x: int, pad_y: int,
                     conf_threshold: float = 0.3) -> List[dict]:
        """Convert raw model output to a list of detections with keypoints."""
        tw, th = self.input_size
        detections = []

        for det in output[0]:
            x1n, y1n, x2n, y2n = det[0:4]
            confidence = det[4]
            if confidence < conf_threshold:
                continue

            # bbox → original coords
            x1 = int((x1n * tw - pad_x) / scale)
            y1 = int((y1n * th - pad_y) / scale)
            x2 = int((x2n * tw - pad_x) / scale)
            y2 = int((y2n * th - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            # keypoints → original coords
            keypoints = []
            for i in range(17):
                idx = 6 + i * 3
                kx = (det[idx] * tw - pad_x) / scale
                ky = (det[idx + 1] * th - pad_y) / scale
                kc = float(det[idx + 2])
                keypoints.append({'x': float(kx), 'y': float(ky), 'confidence': kc})

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(confidence),
                'keypoints': keypoints,
            })

        # keep the most confident detection (primary person)
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def detect(self, frame_bgr: np.ndarray, conf_threshold: float = 0.3) -> List[dict]:
        """Run full detection pipeline on a BGR OpenCV frame.

        Returns list of dicts with 'bbox', 'confidence', and 'keypoints'.
        Each keypoint has 'x', 'y', 'confidence'.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_bgr.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame_rgb)

        self.interpreter.set_tensor(self.input_details[0]['index'], blob)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return self._postprocess(output, orig_w, orig_h, scale, pad_x, pad_y, conf_threshold)


# ── Camera Manager (unchanged) ──

@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    is_external: bool


class CameraManager:
    """Manages camera detection, selection, and switching"""

    EXTERNAL_CAMERA_PATTERNS = [
        "dji", "action", "gopro", "osmo", "insta360",
        "usb", "capture", "cam link", "elgato", "hdmi"
    ]

    def __init__(self, max_cameras: int = 10):
        self.max_cameras = max_cameras
        self.available_cameras: Dict[int, CameraInfo] = {}
        self.current_camera_index: Optional[int] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._auto_detect_thread: Optional[threading.Thread] = None
        self._stop_auto_detect = False

    def detect_cameras(self) -> Dict[int, CameraInfo]:
        self.available_cameras.clear()
        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = self._get_camera_name(i, cap)
                is_external = self._is_external_camera(name, i)
                self.available_cameras[i] = CameraInfo(
                    index=i, name=name, width=width,
                    height=height, is_external=is_external
                )
                cap.release()
        return self.available_cameras

    def _get_camera_name(self, index: int, cap: cv2.VideoCapture) -> str:
        backend = cap.getBackendName()
        if index == 0:
            return f"Built-in Camera ({backend})"
        return f"Camera {index} ({backend})"

    def _is_external_camera(self, name: str, index: int) -> bool:
        name_lower = name.lower()
        for pattern in self.EXTERNAL_CAMERA_PATTERNS:
            if pattern in name_lower:
                return True
        return index > 0

    def get_external_cameras(self) -> List[CameraInfo]:
        return [cam for cam in self.available_cameras.values() if cam.is_external]

    def select_camera(self, index: int) -> bool:
        with self._lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                self.current_camera_index = index
                if index in self.available_cameras and self.available_cameras[index].is_external:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                return True
            self.cap = None
            self.current_camera_index = None
            return False

    def select_best_camera(self) -> bool:
        if not self.available_cameras:
            self.detect_cameras()
        external = self.get_external_cameras()
        if external:
            return self.select_camera(external[0].index)
        if self.available_cameras:
            return self.select_camera(min(self.available_cameras.keys()))
        return False

    def switch_to_next_camera(self) -> Optional[CameraInfo]:
        if not self.available_cameras:
            return None
        indices = sorted(self.available_cameras.keys())
        if self.current_camera_index is None:
            next_index = indices[0]
        else:
            try:
                pos = indices.index(self.current_camera_index)
                next_index = indices[(pos + 1) % len(indices)]
            except ValueError:
                next_index = indices[0]
        if self.select_camera(next_index):
            return self.available_cameras[next_index]
        return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self.cap is None:
                return False, None
            return self.cap.read()

    def start_auto_detect(self, callback, check_interval: float = 2.0):
        self._stop_auto_detect = False

        def detect_loop():
            known = set(self.available_cameras.keys())
            while not self._stop_auto_detect:
                time.sleep(check_interval)
                self.detect_cameras()
                current = set(self.available_cameras.keys())
                for idx in current - known:
                    callback(self.available_cameras[idx], "connected")
                for idx in known - current:
                    callback(CameraInfo(idx, f"Camera {idx}", 0, 0, True), "disconnected")
                known = current

        self._auto_detect_thread = threading.Thread(target=detect_loop, daemon=True)
        self._auto_detect_thread.start()

    def stop_auto_detect(self):
        self._stop_auto_detect = True
        if self._auto_detect_thread:
            self._auto_detect_thread.join(timeout=3.0)

    def get_current_camera_info(self) -> Optional[CameraInfo]:
        if self.current_camera_index is not None:
            return self.available_cameras.get(self.current_camera_index)
        return None

    def release(self):
        self.stop_auto_detect()
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


# ── Guidance enums / dataclasses (unchanged) ──

class SetupStep(Enum):
    DETECTING_PERSON = 1
    CHECK_DISTANCE = 2
    CHECK_ANGLE = 3
    CHECK_VISIBILITY = 4
    SETUP_COMPLETE = 5


class GuidanceStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class GuidanceDirection(Enum):
    NONE = "none"
    GO_UP = "up"
    GO_DOWN = "down"
    GO_LEFT = "left"
    GO_RIGHT = "right"
    GO_CLOSER = "closer"
    GO_FURTHER = "further"


@dataclass
class PositionCheck:
    status: GuidanceStatus
    message: str
    instruction: str
    direction: GuidanceDirection = GuidanceDirection.NONE


@dataclass
class CameraRequirements:
    min_angle: float = 30.0
    max_angle: float = 60.0
    target_angle: float = 45.0
    min_face_size: float = 0.08
    max_face_size: float = 0.25
    face_horizontal_min: float = 0.25
    face_horizontal_max: float = 0.75
    face_vertical_min: float = 0.1
    face_vertical_max: float = 0.5
    min_shoulders_visible: float = 0.7
    min_hands_visible: float = 0.5


# ── Camera Position Guide (YOLO version) ──

class CameraPositionGuide:
    def __init__(self, model_path: str):
        self.estimator = YoloPoseEstimator(model_path)
        self.requirements = CameraRequirements()
        self.current_step = SetupStep.DETECTING_PERSON
        self.setup_confirmed = False
        self.confirmation_frames = 0
        self.required_confirmation_frames = 30

        self.detected_side: Optional[str] = None
        self.smoothed_angle = None
        self.smoothing_factor = 0.3
        self.angle_history: List[float] = []
        self.position_history: List[Tuple[float, float]] = []
        self.history_size = 15

    # ── helpers ──

    def _kp(self, keypoints: List[dict], idx: int):
        """Shortcut to get a keypoint dict by index."""
        return keypoints[idx]

    def _kp_norm(self, keypoints: List[dict], idx: int, img_w: int, img_h: int):
        """Return (x_norm, y_norm, confidence) for a keypoint."""
        kp = keypoints[idx]
        return kp['x'] / img_w, kp['y'] / img_h, kp['confidence']

    def _smooth_value(self, new_val: float, smoothed_val: Optional[float]) -> float:
        if smoothed_val is None:
            return new_val
        return self.smoothing_factor * new_val + (1 - self.smoothing_factor) * smoothed_val

    # ── checks (adapted to COCO keypoints) ──

    def _calculate_body_angle(self, keypoints: List[dict], img_w: int, img_h: int) -> Tuple[float, str]:
        ls = self._kp(keypoints, PoseKeypoint.LEFT_SHOULDER)
        rs = self._kp(keypoints, PoseKeypoint.RIGHT_SHOULDER)

        ls_x_norm = ls['x'] / img_w
        rs_x_norm = rs['x'] / img_w

        # YOLO pose doesn't provide z-depth directly.
        # Estimate viewing angle from shoulder width compression:
        # When viewed from the side, shoulders appear narrower.
        shoulder_width = abs(rs_x_norm - ls_x_norm)

        # Expected frontal shoulder width ~0.25-0.35 of frame for a normal distance.
        # We use the ratio of visible width vs expected width.
        # Also determine side by which shoulder is closer to camera (larger in frame / more centered)
        nose = self._kp(keypoints, PoseKeypoint.NOSE)
        nose_x_norm = nose['x'] / img_w

        # Midpoint of shoulders
        mid_shoulder_x = (ls_x_norm + rs_x_norm) / 2

        # If nose is left of shoulder midpoint → camera on left side
        # If nose is right of shoulder midpoint → camera on right side
        nose_offset = nose_x_norm - mid_shoulder_x

        # Estimate angle from shoulder compression
        # A rough heuristic: frontal view → shoulder_width ~ 0.25+
        # 90° side view → shoulder_width ~ 0.05
        # Map linearly for simplicity
        reference_width = 0.28
        compression_ratio = min(shoulder_width / reference_width, 1.0)
        angle = math.degrees(math.acos(max(0.0, min(1.0, compression_ratio))))

        side = "left" if nose_offset < 0 else "right"

        return angle, side

    def _check_face_position(self, keypoints: List[dict], img_w: int, img_h: int) -> List[PositionCheck]:
        nose_x, nose_y, nose_c = self._kp_norm(keypoints, PoseKeypoint.NOSE, img_w, img_h)

        checks = []
        if nose_x < self.requirements.face_horizontal_min:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR, "Person too far right in frame",
                "Move phone LEFT", GuidanceDirection.GO_LEFT))
        elif nose_x > self.requirements.face_horizontal_max:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR, "Person too far left in frame",
                "Move phone RIGHT", GuidanceDirection.GO_RIGHT))

        if nose_y < self.requirements.face_vertical_min:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR, "Face too high in frame",
                "Raise the phone or tilt it up", GuidanceDirection.GO_DOWN))
        elif nose_y > self.requirements.face_vertical_max:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR, "Face too low in frame",
                "Lower the phone or tilt it down", GuidanceDirection.GO_UP))

        if not checks:
            checks.append(PositionCheck(GuidanceStatus.OK, "Face position OK", ""))
        return checks

    def _check_distance(self, keypoints: List[dict], img_w: int, img_h: int) -> PositionCheck:
        le_x, le_y, le_c = self._kp_norm(keypoints, PoseKeypoint.LEFT_EAR, img_w, img_h)
        re_x, re_y, re_c = self._kp_norm(keypoints, PoseKeypoint.RIGHT_EAR, img_w, img_h)
        ley_x, ley_y, ley_c = self._kp_norm(keypoints, PoseKeypoint.LEFT_EYE, img_w, img_h)
        rey_x, rey_y, rey_c = self._kp_norm(keypoints, PoseKeypoint.RIGHT_EYE, img_w, img_h)
        nose_x, nose_y, nose_c = self._kp_norm(keypoints, PoseKeypoint.NOSE, img_w, img_h)

        ear_dist = abs(le_x - re_x)
        eye_dist = abs(ley_x - rey_x)
        nose_to_eye = abs(nose_y - (ley_y + rey_y) / 2)
        face_size = max(ear_dist, eye_dist * 1.5, nose_to_eye * 2)

        if face_size < self.requirements.min_face_size:
            return PositionCheck(GuidanceStatus.ERROR, "Too far from camera",
                                "Move phone closer to candidate", GuidanceDirection.GO_CLOSER)
        elif face_size > self.requirements.max_face_size:
            return PositionCheck(GuidanceStatus.ERROR, "Too close to camera",
                                "Move phone back from candidate", GuidanceDirection.GO_FURTHER)
        return PositionCheck(GuidanceStatus.OK, "Distance OK", "")

    def _check_angle(self, angle: float, side: str) -> PositionCheck:
        if angle < self.requirements.min_angle:
            return PositionCheck(
                GuidanceStatus.ERROR,
                f"Camera too frontal ({angle:.0f}\u00b0)",
                f"Move phone more to the {side.upper()} side")
        elif angle > self.requirements.max_angle:
            return PositionCheck(
                GuidanceStatus.ERROR,
                f"Camera too far to side ({angle:.0f}\u00b0)",
                "Move phone more towards the FRONT")
        elif abs(angle - self.requirements.target_angle) <= 10:
            return PositionCheck(GuidanceStatus.OK, f"Angle perfect ({angle:.0f}\u00b0)", "")
        else:
            direction = "side" if angle < self.requirements.target_angle else "front"
            return PositionCheck(
                GuidanceStatus.WARNING,
                f"Angle acceptable ({angle:.0f}\u00b0)",
                f"For best results, move slightly towards {direction}")

    def _check_body_visibility(self, keypoints: List[dict]) -> PositionCheck:
        ls_c = keypoints[PoseKeypoint.LEFT_SHOULDER]['confidence']
        rs_c = keypoints[PoseKeypoint.RIGHT_SHOULDER]['confidence']
        shoulder_conf = min(ls_c, rs_c)

        if shoulder_conf < self.requirements.min_shoulders_visible:
            return PositionCheck(GuidanceStatus.ERROR,
                                "Shoulders not clearly visible",
                                "Adjust camera to show both shoulders")

        lw_c = keypoints[PoseKeypoint.LEFT_WRIST]['confidence']
        rw_c = keypoints[PoseKeypoint.RIGHT_WRIST]['confidence']
        best_hand = max(lw_c, rw_c)

        if best_hand < self.requirements.min_hands_visible:
            return PositionCheck(GuidanceStatus.WARNING,
                                "Hands not clearly visible",
                                "Tilt camera DOWN to capture desk area",
                                GuidanceDirection.GO_DOWN)

        return PositionCheck(GuidanceStatus.OK, "Upper body visible", "")

    def _check_stability(self) -> bool:
        if len(self.angle_history) < self.history_size:
            return False
        return np.std(self.angle_history) <= 5.0

    # ── drawing ──

    def _draw_skeleton(self, frame: np.ndarray, keypoints: List[dict]) -> np.ndarray:
        """Draw YOLO pose skeleton on frame."""
        for start_idx, end_idx in SKELETON:
            sk = keypoints[start_idx]
            ek = keypoints[end_idx]
            if sk['confidence'] > KEYPOINT_CONF_THRESHOLD and ek['confidence'] > KEYPOINT_CONF_THRESHOLD:
                pt1 = (int(sk['x']), int(sk['y']))
                pt2 = (int(ek['x']), int(ek['y']))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

        for kp in keypoints:
            if kp['confidence'] > KEYPOINT_CONF_THRESHOLD:
                cv2.circle(frame, (int(kp['x']), int(kp['y'])), 4, KEYPOINT_COLOR, -1)

        return frame

    def _draw_direction_arrow(self, frame: np.ndarray, direction: GuidanceDirection,
                               center_x: int, center_y: int, size: int = 80) -> np.ndarray:
        if direction == GuidanceDirection.NONE:
            return frame

        arrow_color = (0, 165, 255)
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        thickness = 4
        tip_length = 0.4

        arrow_configs = {
            GuidanceDirection.GO_UP: {
                "start": (center_x, center_y - size // 2),
                "end": (center_x, center_y + size // 2),
                "label": "LOWER PHONE",
                "label_pos": (center_x - 65, center_y + size // 2 + 30)
            },
            GuidanceDirection.GO_DOWN: {
                "start": (center_x, center_y + size // 2),
                "end": (center_x, center_y - size // 2),
                "label": "RAISE PHONE",
                "label_pos": (center_x - 60, center_y - size // 2 - 15)
            },
            GuidanceDirection.GO_LEFT: {
                "start": (center_x + size // 2, center_y),
                "end": (center_x - size // 2, center_y),
                "label": "MOVE LEFT",
                "label_pos": (center_x - size // 2 - 90, center_y + 5)
            },
            GuidanceDirection.GO_RIGHT: {
                "start": (center_x - size // 2, center_y),
                "end": (center_x + size // 2, center_y),
                "label": "MOVE RIGHT",
                "label_pos": (center_x + size // 2 + 10, center_y + 5)
            },
            GuidanceDirection.GO_CLOSER: {
                "start": None, "end": None,
                "label": "MOVE CLOSER",
                "label_pos": (center_x - 70, center_y + size // 2 + 30)
            },
            GuidanceDirection.GO_FURTHER: {
                "start": None, "end": None,
                "label": "MOVE BACK",
                "label_pos": (center_x - 55, center_y + size // 2 + 30)
            }
        }

        config = arrow_configs.get(direction)
        if not config:
            return frame

        if direction in [GuidanceDirection.GO_UP, GuidanceDirection.GO_DOWN,
                         GuidanceDirection.GO_LEFT, GuidanceDirection.GO_RIGHT]:
            cv2.arrowedLine(frame, config["start"], config["end"],
                           arrow_color, thickness, tipLength=tip_length)
        elif direction == GuidanceDirection.GO_CLOSER:
            offset = size // 3
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                cv2.arrowedLine(frame,
                    (center_x + dx * offset, center_y + dy * offset),
                    (center_x + dx * offset // 2, center_y + dy * offset // 2),
                    arrow_color, thickness - 1, tipLength=0.5)
        elif direction == GuidanceDirection.GO_FURTHER:
            offset = size // 4
            outer = size // 2
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                cv2.arrowedLine(frame,
                    (center_x + dx * offset, center_y + dy * offset),
                    (center_x + dx * outer, center_y + dy * outer),
                    arrow_color, thickness - 1, tipLength=0.5)

        label = config["label"]
        label_pos = config["label_pos"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        pad = 5
        cv2.rectangle(frame,
                     (label_pos[0] - pad, label_pos[1] - th - pad),
                     (label_pos[0] + tw + pad, label_pos[1] + pad), bg_color, -1)
        cv2.putText(frame, label, label_pos, font, font_scale, text_color, font_thickness)
        return frame

    def _draw_visual_guidance(self, frame: np.ndarray, checks: List[PositionCheck]) -> np.ndarray:
        img_h, img_w = frame.shape[:2]

        directions: List[GuidanceDirection] = []
        for check in checks:
            if check.status == GuidanceStatus.ERROR and check.direction != GuidanceDirection.NONE:
                directions.append(check.direction)
        for check in checks:
            if check.status == GuidanceStatus.WARNING and check.direction != GuidanceDirection.NONE:
                directions.append(check.direction)

        seen = set()
        unique = []
        for d in directions:
            if d not in seen:
                seen.add(d)
                unique.append(d)

        conflicts = [
            {GuidanceDirection.GO_LEFT, GuidanceDirection.GO_RIGHT},
            {GuidanceDirection.GO_UP, GuidanceDirection.GO_DOWN},
            {GuidanceDirection.GO_CLOSER, GuidanceDirection.GO_FURTHER},
        ]
        final = []
        for d in unique:
            conflict = False
            for pair in conflicts:
                if d in pair and (pair - {d}).pop() in final:
                    conflict = True
                    break
            if not conflict:
                final.append(d)

        if not final:
            return frame

        base_x = img_w - 120
        base_y = img_h // 2
        if len(final) == 1:
            positions = {final[0]: (base_x, base_y)}
        else:
            spacing = 120
            total = (len(final) - 1) * spacing
            start_y = base_y - total // 2
            positions = {d: (base_x, start_y + i * spacing) for i, d in enumerate(final)}

        for d in final:
            cx, cy = positions[d]
            frame = self._draw_direction_arrow(frame, d, cx, cy, size=80)
        return frame

    def _draw_guidance_overlay(self, frame: np.ndarray, checks: List[PositionCheck],
                                angle: float, side: str) -> np.ndarray:
        img_h, img_w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        step_names = {
            SetupStep.DETECTING_PERSON: "Step 1: Detecting Person...",
            SetupStep.CHECK_DISTANCE: "Step 2: Checking Distance",
            SetupStep.CHECK_ANGLE: "Step 3: Checking Camera Angle",
            SetupStep.CHECK_VISIBILITY: "Step 4: Checking Visibility",
            SetupStep.SETUP_COMPLETE: "Setup Complete!"
        }
        title = step_names.get(self.current_step, "Camera Setup")
        cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if side:
            cv2.putText(frame, f"Camera on {side.upper()} side | Angle: {abs(angle):.0f}deg",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y_pos = 95
        for check in checks:
            if check.status == GuidanceStatus.OK:
                color, icon = (0, 255, 0), "[OK]"
            elif check.status == GuidanceStatus.WARNING:
                color, icon = (0, 200, 255), "[!]"
            else:
                color, icon = (0, 0, 255), "[X]"

            cv2.putText(frame, f"{icon} {check.message}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if check.instruction:
                cv2.putText(frame, f"    -> {check.instruction}", (20, y_pos + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                y_pos += 45
            else:
                y_pos += 25

        if self.current_step == SetupStep.SETUP_COMPLETE or self._check_stability():
            progress = min(1.0, self.confirmation_frames / self.required_confirmation_frames)
            bar_w = int((img_w - 40) * progress)
            cv2.rectangle(frame, (20, img_h - 60), (img_w - 20, img_h - 40), (100, 100, 100), 2)
            if bar_w > 0:
                cv2.rectangle(frame, (22, img_h - 58), (22 + bar_w, img_h - 42), (0, 255, 0), -1)
            if self.setup_confirmed:
                cv2.putText(frame, "POSITION CONFIRMED - Ready for proctoring!",
                           (20, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Hold steady to confirm position...",
                           (20, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, "R: Restart | C: Switch Camera | Q: Quit",
                   (20, img_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        frame = self._draw_visual_guidance(frame, checks)
        return frame

    def draw_camera_info(self, frame: np.ndarray, camera_info: Optional[CameraInfo],
                         notification: str = "") -> np.ndarray:
        img_h, img_w = frame.shape[:2]
        if camera_info:
            cam_text = f"CAM: {camera_info.name}"
            external_badge = " [EXT]" if camera_info.is_external else ""
            res_text = f"{camera_info.width}x{camera_info.height}"
            text_size = cv2.getTextSize(cam_text + external_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_pos = img_w - text_size[0] - 20
            cv2.putText(frame, cam_text + external_badge, (x_pos, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, res_text, (x_pos, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        if notification:
            cv2.rectangle(frame, (0, img_h // 2 - 30), (img_w, img_h // 2 + 30), (50, 50, 50), -1)
            text_size = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (img_w - text_size[0]) // 2
            cv2.putText(frame, notification, (text_x, img_h // 2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame

    def _draw_position_guides(self, frame: np.ndarray, keypoints: Optional[List[dict]]) -> np.ndarray:
        img_h, img_w = frame.shape[:2]
        x1 = int(self.requirements.face_horizontal_min * img_w)
        x2 = int(self.requirements.face_horizontal_max * img_w)
        y1 = int(self.requirements.face_vertical_min * img_h)
        y2 = int(self.requirements.face_vertical_max * img_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        if keypoints:
            nose = keypoints[PoseKeypoint.NOSE]
            nose_x, nose_y = int(nose['x']), int(nose['y'])
            in_zone = (x1 <= nose_x <= x2) and (y1 <= nose_y <= y2)
            color = (0, 255, 0) if in_zone else (0, 0, 255)
            cv2.circle(frame, (nose_x, nose_y), 10, color, -1)
            cv2.circle(frame, (nose_x, nose_y), 15, color, 2)
        return frame

    # ── main processing ──

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a frame and return annotated frame with guidance."""
        img_h, img_w = frame.shape[:2]
        checks = []
        angle = 0.0
        side = ""

        detections = self.estimator.detect(frame, conf_threshold=0.3)

        if not detections:
            self.current_step = SetupStep.DETECTING_PERSON
            self.confirmation_frames = 0
            checks.append(PositionCheck(
                GuidanceStatus.ERROR, "No person detected",
                "Ensure candidate is visible in camera"))
            return self._draw_guidance_overlay(frame, checks, angle, side), False

        # Use the most confident person detection
        best = detections[0]
        keypoints = best['keypoints']

        # Draw skeleton
        frame = self._draw_skeleton(frame, keypoints)

        # Draw position guides
        frame = self._draw_position_guides(frame, keypoints)

        # Calculate angle
        raw_angle, side = self._calculate_body_angle(keypoints, img_w, img_h)
        self.smoothed_angle = self._smooth_value(raw_angle, self.smoothed_angle)
        angle = self.smoothed_angle
        self.detected_side = side

        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)

        # Run all checks
        distance_check = self._check_distance(keypoints, img_w, img_h)
        angle_check = self._check_angle(angle, side)
        position_checks = self._check_face_position(keypoints, img_w, img_h)
        visibility_check = self._check_body_visibility(keypoints)

        checks = [distance_check, angle_check] + position_checks + [visibility_check]

        all_ok = all(c.status != GuidanceStatus.ERROR for c in checks)
        position_has_error = any(c.status == GuidanceStatus.ERROR for c in position_checks)

        if distance_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_DISTANCE
        elif angle_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_ANGLE
        elif position_has_error or visibility_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_VISIBILITY
        elif all_ok:
            self.current_step = SetupStep.SETUP_COMPLETE

        if all_ok and self._check_stability():
            self.confirmation_frames += 1
            if self.confirmation_frames >= self.required_confirmation_frames:
                self.setup_confirmed = True
        else:
            self.confirmation_frames = max(0, self.confirmation_frames - 2)
            self.setup_confirmed = False

        frame = self._draw_guidance_overlay(frame, checks, angle, side)
        return frame, self.setup_confirmed

    def reset(self):
        self.current_step = SetupStep.DETECTING_PERSON
        self.setup_confirmed = False
        self.confirmation_frames = 0
        self.smoothed_angle = None
        self.angle_history.clear()
        self.position_history.clear()


# ── Main ──

def main():
    # Resolve model path relative to this script
    script_dir = Path(__file__).resolve().parent
    model_path = str(script_dir / "pose_object_detection_running" / "saved_models"
                     / "yolo11n-pose_saved_model" / "yolo11n-pose_float16.tflite")

    print(f"Loading YOLO pose model: {model_path}")
    guide = CameraPositionGuide(model_path)
    camera_manager = CameraManager()

    notification_text = ""
    notification_expire_time = 0

    def on_camera_change(camera_info: CameraInfo, event: str):
        nonlocal notification_text, notification_expire_time
        if event == "connected":
            notification_text = f"Camera connected: {camera_info.name}"
            print(f"\n[AUTO-DETECT] New camera connected: {camera_info.name}")
            if camera_info.is_external:
                camera_manager.select_camera(camera_info.index)
                notification_text = f"Switched to: {camera_info.name}"
                print(f"[AUTO-SWITCH] Switched to external camera: {camera_info.name}")
        else:
            notification_text = f"Camera disconnected: {camera_info.name}"
            print(f"\n[AUTO-DETECT] Camera disconnected: {camera_info.name}")
            if camera_info.index == camera_manager.current_camera_index:
                camera_manager.detect_cameras()
                if camera_manager.select_best_camera():
                    new_cam = camera_manager.get_current_camera_info()
                    notification_text = f"Switched to: {new_cam.name if new_cam else 'Unknown'}"
        notification_expire_time = time.time() + 3.0

    print("=" * 60)
    print("CAMERA POSITION GUIDE FOR AI PROCTORING (YOLO)")
    print("=" * 60)

    print("\nScanning for cameras...")
    cameras = camera_manager.detect_cameras()
    if not cameras:
        print("Error: No cameras found!")
        return

    print(f"\nFound {len(cameras)} camera(s):")
    for idx, cam in cameras.items():
        ext_tag = " [EXTERNAL]" if cam.is_external else ""
        print(f"  [{idx}] {cam.name} ({cam.width}x{cam.height}){ext_tag}")

    if camera_manager.select_best_camera():
        current = camera_manager.get_current_camera_info()
        print(f"\nSelected: {current.name if current else 'Unknown'}")
    else:
        print("Error: Could not open any camera")
        return

    camera_manager.start_auto_detect(on_camera_change, check_interval=2.0)
    print("\n[AUTO-DETECT] Monitoring for camera changes...")

    print("\nREQUIREMENTS:")
    print("  - Phone/camera on LEFT or RIGHT side of candidate")
    print("  - 45 degree angle (diagonal view)")
    print("  - Upper body and desk visible")
    print("\nCONTROLS:")
    print("  'C' - Switch to next camera")
    print("  'R' - Restart setup")
    print("  'Q' - Quit")
    print("=" * 60)

    while True:
        ret, frame = camera_manager.read_frame()
        if not ret:
            print("Warning: Failed to read frame, retrying...")
            time.sleep(0.5)
            camera_manager.detect_cameras()
            if camera_manager.select_best_camera():
                continue
            else:
                print("Error: No cameras available")
                break

        processed_frame, is_complete = guide.process_frame(frame)

        current_cam = camera_manager.get_current_camera_info()
        current_notification = notification_text if time.time() < notification_expire_time else ""
        processed_frame = guide.draw_camera_info(processed_frame, current_cam, current_notification)

        cv2.imshow('Camera Position Guide - AI Proctoring (YOLO)', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            guide.reset()
            print("Setup reset - starting over...")
        elif key == ord('c') or key == ord('C'):
            new_cam = camera_manager.switch_to_next_camera()
            if new_cam:
                print(f"Switched to: {new_cam.name}")
                notification_text = f"Switched to: {new_cam.name}"
                notification_expire_time = time.time() + 2.0
                guide.reset()
            else:
                print("No other cameras available")

    camera_manager.release()
    cv2.destroyAllWindows()

    if guide.setup_confirmed:
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        current = camera_manager.get_current_camera_info()
        if current:
            print(f"Camera: {current.name}")
        print(f"Position: {guide.detected_side.upper()} side")
        print("Ready to begin proctoring session.")
        print("=" * 60)


if __name__ == "__main__":
    main()
