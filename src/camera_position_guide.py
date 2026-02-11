"""
Camera Position Guide for AI Proctoring

Guides candidates to properly position their phone camera at a 45° side angle
to capture upper body and desk workspace.

Requirements:
- Phone placed on either side of candidate
- 45° angle view (diagonal showing profile and desk)
- Upper body visible (face, shoulders, hands)
- Desk/workspace visible
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    is_external: bool


class CameraManager:
    """Manages camera detection, selection, and switching"""

    # Common external camera name patterns
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
        """Scan for all available cameras"""
        self.available_cameras.clear()

        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Try to get camera name (platform dependent)
                name = self._get_camera_name(i, cap)

                # Check if it's likely an external camera
                is_external = self._is_external_camera(name, i)

                self.available_cameras[i] = CameraInfo(
                    index=i,
                    name=name,
                    width=width,
                    height=height,
                    is_external=is_external
                )
                cap.release()

        return self.available_cameras

    def _get_camera_name(self, index: int, cap: cv2.VideoCapture) -> str:
        """Try to get camera name - falls back to generic name"""
        # OpenCV doesn't provide camera names directly on all platforms
        # On macOS, we can try to identify by index patterns
        backend = cap.getBackendName()

        # Default naming based on index
        if index == 0:
            return f"Built-in Camera ({backend})"
        else:
            return f"Camera {index} ({backend})"

    def _is_external_camera(self, name: str, index: int) -> bool:
        """Determine if camera is likely external"""
        name_lower = name.lower()

        # Check name patterns
        for pattern in self.EXTERNAL_CAMERA_PATTERNS:
            if pattern in name_lower:
                return True

        # Index > 0 is often external on laptops
        if index > 0:
            return True

        return False

    def get_external_cameras(self) -> List[CameraInfo]:
        """Get list of external cameras only"""
        return [cam for cam in self.available_cameras.values() if cam.is_external]

    def select_camera(self, index: int) -> bool:
        """Select and open a specific camera"""
        with self._lock:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                self.current_camera_index = index
                # Set preferred resolution for external cameras
                if index in self.available_cameras and self.available_cameras[index].is_external:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                return True
            else:
                self.cap = None
                self.current_camera_index = None
                return False

    def select_best_camera(self) -> bool:
        """Auto-select the best camera (prefers external cameras)"""
        if not self.available_cameras:
            self.detect_cameras()

        # Prefer external cameras (like DJI Action Cam)
        external = self.get_external_cameras()
        if external:
            return self.select_camera(external[0].index)

        # Fall back to first available
        if self.available_cameras:
            first_index = min(self.available_cameras.keys())
            return self.select_camera(first_index)

        return False

    def switch_to_next_camera(self) -> Optional[CameraInfo]:
        """Switch to next available camera"""
        if not self.available_cameras:
            return None

        indices = sorted(self.available_cameras.keys())
        if self.current_camera_index is None:
            next_index = indices[0]
        else:
            try:
                current_pos = indices.index(self.current_camera_index)
                next_index = indices[(current_pos + 1) % len(indices)]
            except ValueError:
                next_index = indices[0]

        if self.select_camera(next_index):
            return self.available_cameras[next_index]
        return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from current camera"""
        with self._lock:
            if self.cap is None:
                return False, None
            return self.cap.read()

    def start_auto_detect(self, callback, check_interval: float = 2.0):
        """Start background thread to detect new cameras"""
        self._stop_auto_detect = False

        def detect_loop():
            known_cameras = set(self.available_cameras.keys())
            while not self._stop_auto_detect:
                time.sleep(check_interval)
                self.detect_cameras()
                current_cameras = set(self.available_cameras.keys())

                # Check for new cameras
                new_cameras = current_cameras - known_cameras
                if new_cameras:
                    for idx in new_cameras:
                        callback(self.available_cameras[idx], "connected")

                # Check for disconnected cameras
                removed_cameras = known_cameras - current_cameras
                if removed_cameras:
                    for idx in removed_cameras:
                        callback(CameraInfo(idx, f"Camera {idx}", 0, 0, True), "disconnected")

                known_cameras = current_cameras

        self._auto_detect_thread = threading.Thread(target=detect_loop, daemon=True)
        self._auto_detect_thread.start()

    def stop_auto_detect(self):
        """Stop the auto-detection thread"""
        self._stop_auto_detect = True
        if self._auto_detect_thread:
            self._auto_detect_thread.join(timeout=3.0)

    def get_current_camera_info(self) -> Optional[CameraInfo]:
        """Get info about currently selected camera"""
        if self.current_camera_index is not None:
            return self.available_cameras.get(self.current_camera_index)
        return None

    def release(self):
        """Release camera resources"""
        self.stop_auto_detect()
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


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
    """Visual guidance directions for camera adjustment"""
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
    # Angle requirements (degrees from frontal view)
    min_angle: float = 30.0  # Minimum side angle
    max_angle: float = 60.0  # Maximum side angle
    target_angle: float = 45.0  # Ideal angle

    # Visibility requirements (as ratio of frame)
    min_face_size: float = 0.08  # Face should be at least 8% of frame
    max_face_size: float = 0.25  # Face shouldn't be more than 25%

    # Position requirements (normalized 0-1)
    face_horizontal_min: float = 0.25  # Face should be in middle 50% horizontally
    face_horizontal_max: float = 0.75
    face_vertical_min: float = 0.1  # Face in upper portion
    face_vertical_max: float = 0.5

    # Body visibility
    min_shoulders_visible: float = 0.7  # At least 70% confidence on shoulders
    min_hands_visible: float = 0.5  # At least 50% confidence on hands


class CameraPositionGuide:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True
        )

        self.requirements = CameraRequirements()
        self.current_step = SetupStep.DETECTING_PERSON
        self.setup_confirmed = False
        self.confirmation_frames = 0
        self.required_confirmation_frames = 30  # ~1 second at 30fps

        # Detected side (left or right)
        self.detected_side: Optional[str] = None

        # Smoothing for stability
        self.smoothed_angle = None
        self.smoothing_factor = 0.3

        # History for stability check
        self.angle_history: List[float] = []
        self.position_history: List[Tuple[float, float]] = []
        self.history_size = 15

    def _smooth_value(self, new_val: float, smoothed_val: Optional[float]) -> float:
        if smoothed_val is None:
            return new_val
        return self.smoothing_factor * new_val + (1 - self.smoothing_factor) * smoothed_val

    def _calculate_body_angle(self, landmarks, img_w: int, img_h: int) -> Tuple[float, str]:
        """
        Calculate the angle at which we're viewing the person.
        Returns angle in degrees and detected side.

        Positive angle = viewing from right side
        Negative angle = viewing from left side
        """
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Z-depth difference indicates rotation
        z_diff = left_shoulder.z - right_shoulder.z

        # Shoulder width ratio (compressed when turned)
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)

        # Calculate angle from z-depth and width compression
        angle = math.degrees(math.atan2(z_diff, shoulder_width + 0.01))

        # Determine which side the camera is on
        # If left shoulder appears closer (smaller z), camera is on left side
        side = "right" if z_diff > 0 else "left"

        return angle, side

    def _check_face_position(self, landmarks, img_w: int, img_h: int) -> List[PositionCheck]:
        """Check if face is properly positioned in frame.
        Returns a list of checks so multiple issues can be reported simultaneously.
        """
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        # Face center position (normalized)
        face_x = nose.x
        face_y = nose.y

        # Approximate face size from ear distance
        face_width = abs(left_ear.x - right_ear.x)

        checks = []

        # Check horizontal position
        if face_x < self.requirements.face_horizontal_min:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR,
                "Person too far right in frame",
                "Move phone LEFT",
                GuidanceDirection.GO_LEFT
            ))
        elif face_x > self.requirements.face_horizontal_max:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR,
                "Person too far left in frame",
                "Move phone RIGHT",
                GuidanceDirection.GO_RIGHT
            ))

        # Check vertical position
        if face_y < self.requirements.face_vertical_min:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR,
                "Face too high in frame",
                "Raise the phone or tilt it up",
                GuidanceDirection.GO_DOWN
            ))
        elif face_y > self.requirements.face_vertical_max:
            checks.append(PositionCheck(
                GuidanceStatus.ERROR,
                "Face too low in frame",
                "Lower the phone or tilt it down",
                GuidanceDirection.GO_UP
            ))

        # If no issues, return OK
        if not checks:
            checks.append(PositionCheck(
                GuidanceStatus.OK,
                "Face position OK",
                "",
                GuidanceDirection.NONE
            ))

        return checks

    def _check_distance(self, landmarks, img_w: int, img_h: int) -> PositionCheck:
        """Check if person is at correct distance from camera using face size"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]

        # Calculate face size using multiple measurements for robustness
        # Ear to ear distance (works well for frontal/slight angle views)
        ear_distance = abs(left_ear.x - right_ear.x)

        # Eye to eye distance (more stable when face is partially turned)
        eye_distance = abs(left_eye.x - right_eye.x)

        # Nose to eye vertical distance (helps estimate face height)
        nose_to_eye = abs(nose.y - (left_eye.y + right_eye.y) / 2)

        # Use the maximum of these measurements as face size indicator
        # This handles different viewing angles better
        face_size = max(ear_distance, eye_distance * 1.5, nose_to_eye * 2)

        if face_size < self.requirements.min_face_size:
            return PositionCheck(
                GuidanceStatus.ERROR,
                "Too far from camera",
                "Move phone closer to candidate",
                GuidanceDirection.GO_CLOSER
            )
        elif face_size > self.requirements.max_face_size:
            return PositionCheck(
                GuidanceStatus.ERROR,
                "Too close to camera",
                "Move phone back from candidate",
                GuidanceDirection.GO_FURTHER
            )

        return PositionCheck(
            GuidanceStatus.OK,
            "Distance OK",
            "",
            GuidanceDirection.NONE
        )

    def _check_angle(self, angle: float, side: str) -> PositionCheck:
        """Check if camera is at correct 45° angle"""
        abs_angle = abs(angle)

        if abs_angle < self.requirements.min_angle:
            return PositionCheck(
                GuidanceStatus.ERROR,
                f"Camera too frontal ({abs_angle:.0f}°)",
                f"Move phone more to the {side.upper()} side",
                GuidanceDirection.NONE  # Angle adjustment is more complex
            )
        elif abs_angle > self.requirements.max_angle:
            return PositionCheck(
                GuidanceStatus.ERROR,
                f"Camera too far to side ({abs_angle:.0f}°)",
                "Move phone more towards the FRONT",
                GuidanceDirection.NONE
            )
        elif abs(abs_angle - self.requirements.target_angle) <= 10:
            return PositionCheck(
                GuidanceStatus.OK,
                f"Angle perfect ({abs_angle:.0f}°)",
                "",
                GuidanceDirection.NONE
            )
        else:
            direction = "side" if abs_angle < self.requirements.target_angle else "front"
            return PositionCheck(
                GuidanceStatus.WARNING,
                f"Angle acceptable ({abs_angle:.0f}°)",
                f"For best results, move slightly towards {direction}",
                GuidanceDirection.NONE
            )

    def _check_body_visibility(self, landmarks) -> PositionCheck:
        """Check if upper body parts are visible"""
        # Check shoulder visibility
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        shoulder_conf = min(left_shoulder.visibility, right_shoulder.visibility)

        if shoulder_conf < self.requirements.min_shoulders_visible:
            return PositionCheck(
                GuidanceStatus.ERROR,
                "Shoulders not clearly visible",
                "Adjust camera to show both shoulders",
                GuidanceDirection.NONE
            )

        # Check hand/wrist visibility
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # At least one hand should be visible (candidate writing)
        best_hand_conf = max(left_wrist.visibility, right_wrist.visibility)

        if best_hand_conf < self.requirements.min_hands_visible:
            return PositionCheck(
                GuidanceStatus.WARNING,
                "Hands not clearly visible",
                "Tilt camera DOWN to capture desk area",
                GuidanceDirection.GO_DOWN
            )

        return PositionCheck(
            GuidanceStatus.OK,
            "Upper body visible",
            "",
            GuidanceDirection.NONE
        )

    def _check_stability(self) -> bool:
        """Check if position has been stable for confirmation"""
        if len(self.angle_history) < self.history_size:
            return False

        # Check angle stability
        angle_std = np.std(self.angle_history)
        if angle_std > 5.0:  # More than 5 degree variation
            return False

        return True

    def _draw_direction_arrow(self, frame: np.ndarray, direction: GuidanceDirection,
                               center_x: int, center_y: int, size: int = 80) -> np.ndarray:
        """Draw a directional arrow with label on the frame"""
        if direction == GuidanceDirection.NONE:
            return frame

        # Arrow colors (bright for visibility)
        arrow_color = (0, 165, 255)  # Orange
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background

        # Arrow thickness
        thickness = 4
        tip_length = 0.4

        # Define arrow endpoints and labels based on direction
        # Labels describe what to do with the PHONE/CAMERA position
        arrow_configs = {
            GuidanceDirection.GO_UP: {
                # Face too low → move phone DOWN to bring face up in frame
                "start": (center_x, center_y - size // 2),
                "end": (center_x, center_y + size // 2),
                "label": "LOWER PHONE",
                "label_pos": (center_x - 65, center_y + size // 2 + 30)
            },
            GuidanceDirection.GO_DOWN: {
                # Face too high → move phone UP to bring face down in frame
                "start": (center_x, center_y + size // 2),
                "end": (center_x, center_y - size // 2),
                "label": "RAISE PHONE",
                "label_pos": (center_x - 60, center_y - size // 2 - 15)
            },
            GuidanceDirection.GO_LEFT: {
                # Person too far right → move phone LEFT
                "start": (center_x + size // 2, center_y),
                "end": (center_x - size // 2, center_y),
                "label": "MOVE LEFT",
                "label_pos": (center_x - size // 2 - 90, center_y + 5)
            },
            GuidanceDirection.GO_RIGHT: {
                # Person too far left → move phone RIGHT
                "start": (center_x - size // 2, center_y),
                "end": (center_x + size // 2, center_y),
                "label": "MOVE RIGHT",
                "label_pos": (center_x + size // 2 + 10, center_y + 5)
            },
            GuidanceDirection.GO_CLOSER: {
                "start": None,  # Special case - draw zoom in icon
                "end": None,
                "label": "MOVE CLOSER",
                "label_pos": (center_x - 70, center_y + size // 2 + 30)
            },
            GuidanceDirection.GO_FURTHER: {
                "start": None,  # Special case - draw zoom out icon
                "end": None,
                "label": "MOVE BACK",
                "label_pos": (center_x - 55, center_y + size // 2 + 30)
            }
        }

        config = arrow_configs.get(direction)
        if not config:
            return frame

        # Draw the arrow or special icon
        if direction in [GuidanceDirection.GO_UP, GuidanceDirection.GO_DOWN,
                         GuidanceDirection.GO_LEFT, GuidanceDirection.GO_RIGHT]:
            # Draw arrow
            cv2.arrowedLine(frame, config["start"], config["end"],
                           arrow_color, thickness, tipLength=tip_length)
        elif direction == GuidanceDirection.GO_CLOSER:
            # Draw zoom in icon (converging arrows)
            offset = size // 3
            # Top-left arrow pointing to center
            cv2.arrowedLine(frame, (center_x - offset, center_y - offset),
                           (center_x - offset // 2, center_y - offset // 2),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Top-right arrow pointing to center
            cv2.arrowedLine(frame, (center_x + offset, center_y - offset),
                           (center_x + offset // 2, center_y - offset // 2),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Bottom-left arrow pointing to center
            cv2.arrowedLine(frame, (center_x - offset, center_y + offset),
                           (center_x - offset // 2, center_y + offset // 2),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Bottom-right arrow pointing to center
            cv2.arrowedLine(frame, (center_x + offset, center_y + offset),
                           (center_x + offset // 2, center_y + offset // 2),
                           arrow_color, thickness - 1, tipLength=0.5)
        elif direction == GuidanceDirection.GO_FURTHER:
            # Draw zoom out icon (diverging arrows)
            offset = size // 4
            outer_offset = size // 2
            # Center to top-left
            cv2.arrowedLine(frame, (center_x - offset, center_y - offset),
                           (center_x - outer_offset, center_y - outer_offset),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Center to top-right
            cv2.arrowedLine(frame, (center_x + offset, center_y - offset),
                           (center_x + outer_offset, center_y - outer_offset),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Center to bottom-left
            cv2.arrowedLine(frame, (center_x - offset, center_y + offset),
                           (center_x - outer_offset, center_y + outer_offset),
                           arrow_color, thickness - 1, tipLength=0.5)
            # Center to bottom-right
            cv2.arrowedLine(frame, (center_x + offset, center_y + offset),
                           (center_x + outer_offset, center_y + outer_offset),
                           arrow_color, thickness - 1, tipLength=0.5)

        # Draw label with background
        label = config["label"]
        label_pos = config["label_pos"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        padding = 5

        # Draw background rectangle
        cv2.rectangle(frame,
                     (label_pos[0] - padding, label_pos[1] - text_h - padding),
                     (label_pos[0] + text_w + padding, label_pos[1] + padding),
                     bg_color, -1)

        # Draw text
        cv2.putText(frame, label, label_pos, font, font_scale, text_color, font_thickness)

        return frame

    def _draw_visual_guidance(self, frame: np.ndarray, checks: List[PositionCheck]) -> np.ndarray:
        """Draw visual guidance arrows based on position checks.

        Can display multiple non-conflicting directions simultaneously.
        Conflicting pairs (cannot show together):
        - GO_LEFT and GO_RIGHT
        - GO_UP and GO_DOWN
        - GO_CLOSER and GO_FURTHER
        """
        img_h, img_w = frame.shape[:2]

        # Collect all directions from errors first, then warnings
        directions_to_show: List[GuidanceDirection] = []

        # First pass: collect error directions
        for check in checks:
            if check.status == GuidanceStatus.ERROR and check.direction != GuidanceDirection.NONE:
                directions_to_show.append(check.direction)

        # Second pass: collect warning directions
        for check in checks:
            if check.status == GuidanceStatus.WARNING and check.direction != GuidanceDirection.NONE:
                directions_to_show.append(check.direction)

        # Remove duplicates while preserving order
        seen = set()
        unique_directions = []
        for d in directions_to_show:
            if d not in seen:
                seen.add(d)
                unique_directions.append(d)

        # Define conflicting pairs
        conflicts = [
            {GuidanceDirection.GO_LEFT, GuidanceDirection.GO_RIGHT},
            {GuidanceDirection.GO_UP, GuidanceDirection.GO_DOWN},
            {GuidanceDirection.GO_CLOSER, GuidanceDirection.GO_FURTHER},
        ]

        # Filter out conflicting directions (keep the first one encountered)
        final_directions = []
        for direction in unique_directions:
            has_conflict = False
            for conflict_pair in conflicts:
                if direction in conflict_pair:
                    # Check if we already have the conflicting direction
                    other = (conflict_pair - {direction}).pop()
                    if other in final_directions:
                        has_conflict = True
                        break
            if not has_conflict:
                final_directions.append(direction)

        if not final_directions:
            return frame

        # Define positions for different directions to avoid overlap
        # Position arrows around the right side of the frame
        base_x = img_w - 120
        base_y = img_h // 2

        # Calculate positions based on which directions we need to show
        positions = {}

        # Calculate positions to spread arrows out vertically
        if len(final_directions) == 1:
            positions[final_directions[0]] = (base_x, base_y)
        else:
            # Spread multiple arrows vertically
            spacing = 120
            total_height = (len(final_directions) - 1) * spacing
            start_y = base_y - total_height // 2

            for i, direction in enumerate(final_directions):
                positions[direction] = (base_x, start_y + i * spacing)

        # Draw all non-conflicting arrows
        for direction in final_directions:
            center_x, center_y = positions[direction]
            frame = self._draw_direction_arrow(frame, direction, center_x, center_y, size=80)

        return frame

    def _draw_guidance_overlay(self, frame: np.ndarray, checks: List[PositionCheck],
                                angle: float, side: str) -> np.ndarray:
        """Draw guidance information on frame"""
        img_h, img_w = frame.shape[:2]

        # Semi-transparent overlay for text area
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        step_names = {
            SetupStep.DETECTING_PERSON: "Step 1: Detecting Person...",
            SetupStep.CHECK_DISTANCE: "Step 2: Checking Distance",
            SetupStep.CHECK_ANGLE: "Step 3: Checking Camera Angle",
            SetupStep.CHECK_VISIBILITY: "Step 4: Checking Visibility",
            SetupStep.SETUP_COMPLETE: "Setup Complete!"
        }

        title = step_names.get(self.current_step, "Camera Setup")
        cv2.putText(frame, title, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Current angle and side
        if side:
            side_text = f"Camera on {side.upper()} side | Angle: {abs(angle):.0f}deg"
            cv2.putText(frame, side_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw checks
        y_pos = 95
        for check in checks:
            if check.status == GuidanceStatus.OK:
                color = (0, 255, 0)
                icon = "[OK]"
            elif check.status == GuidanceStatus.WARNING:
                color = (0, 200, 255)
                icon = "[!]"
            else:
                color = (0, 0, 255)
                icon = "[X]"

            cv2.putText(frame, f"{icon} {check.message}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if check.instruction:
                cv2.putText(frame, f"    -> {check.instruction}", (20, y_pos + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                y_pos += 45
            else:
                y_pos += 25

        # Progress bar for confirmation
        if self.current_step == SetupStep.SETUP_COMPLETE or self._check_stability():
            progress = min(1.0, self.confirmation_frames / self.required_confirmation_frames)
            bar_width = int((img_w - 40) * progress)

            cv2.rectangle(frame, (20, img_h - 60), (img_w - 20, img_h - 40), (100, 100, 100), 2)
            if bar_width > 0:
                cv2.rectangle(frame, (22, img_h - 58), (22 + bar_width, img_h - 42), (0, 255, 0), -1)

            if self.setup_confirmed:
                cv2.putText(frame, "POSITION CONFIRMED - Ready for proctoring!",
                           (20, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Hold steady to confirm position...",
                           (20, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Instructions at bottom
        cv2.putText(frame, "R: Restart | C: Switch Camera | Q: Quit",
                   (20, img_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Draw visual guidance arrows
        frame = self._draw_visual_guidance(frame, checks)

        return frame

    def draw_camera_info(self, frame: np.ndarray, camera_info: Optional[CameraInfo],
                         notification: str = "") -> np.ndarray:
        """Draw camera information on frame"""
        img_h, img_w = frame.shape[:2]

        if camera_info:
            # Camera info in top-right corner
            cam_text = f"CAM: {camera_info.name}"
            res_text = f"{camera_info.width}x{camera_info.height}"
            external_badge = " [EXT]" if camera_info.is_external else ""

            text_size = cv2.getTextSize(cam_text + external_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x_pos = img_w - text_size[0] - 20

            cv2.putText(frame, cam_text + external_badge, (x_pos, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, res_text, (x_pos, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Notification banner (for camera connect/disconnect)
        if notification:
            # Draw notification banner at top
            cv2.rectangle(frame, (0, img_h // 2 - 30), (img_w, img_h // 2 + 30), (50, 50, 50), -1)
            text_size = cv2.getTextSize(notification, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (img_w - text_size[0]) // 2
            cv2.putText(frame, notification, (text_x, img_h // 2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def _draw_position_guides(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw visual guides showing ideal positioning zones"""
        img_h, img_w = frame.shape[:2]

        # Draw ideal face zone (semi-transparent rectangle)
        x1 = int(self.requirements.face_horizontal_min * img_w)
        x2 = int(self.requirements.face_horizontal_max * img_w)
        y1 = int(self.requirements.face_vertical_min * img_h)
        y2 = int(self.requirements.face_vertical_max * img_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw current face position indicator
        if landmarks:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * img_w)
            nose_y = int(nose.y * img_h)

            # Check if in zone
            in_zone = (x1 <= nose_x <= x2) and (y1 <= nose_y <= y2)
            color = (0, 255, 0) if in_zone else (0, 0, 255)

            cv2.circle(frame, (nose_x, nose_y), 10, color, -1)
            cv2.circle(frame, (nose_x, nose_y), 15, color, 2)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a frame and return annotated frame with guidance.
        Returns (annotated_frame, is_setup_complete)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        img_h, img_w = frame.shape[:2]
        checks = []
        angle = 0.0
        side = ""

        if not results.pose_landmarks:
            self.current_step = SetupStep.DETECTING_PERSON
            self.confirmation_frames = 0
            checks.append(PositionCheck(
                GuidanceStatus.ERROR,
                "No person detected",
                "Ensure candidate is visible in camera",
                GuidanceDirection.NONE
            ))
            return self._draw_guidance_overlay(frame, checks, angle, side), False

        landmarks = results.pose_landmarks.landmark

        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw position guides
        frame = self._draw_position_guides(frame, landmarks)

        # Calculate angle
        raw_angle, side = self._calculate_body_angle(landmarks, img_w, img_h)
        self.smoothed_angle = self._smooth_value(raw_angle, self.smoothed_angle)
        angle = self.smoothed_angle
        self.detected_side = side

        # Update history
        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)

        # Run all checks
        distance_check = self._check_distance(landmarks, img_w, img_h)
        angle_check = self._check_angle(angle, side)
        position_checks = self._check_face_position(landmarks, img_w, img_h)  # Returns list
        visibility_check = self._check_body_visibility(landmarks)

        # Combine all checks into a single list
        checks = [distance_check, angle_check] + position_checks + [visibility_check]

        # Determine current step based on checks
        all_ok = all(c.status != GuidanceStatus.ERROR for c in checks)

        # Check if any position check has an error
        position_has_error = any(c.status == GuidanceStatus.ERROR for c in position_checks)

        if distance_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_DISTANCE
        elif angle_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_ANGLE
        elif position_has_error or visibility_check.status == GuidanceStatus.ERROR:
            self.current_step = SetupStep.CHECK_VISIBILITY
        elif all_ok:
            self.current_step = SetupStep.SETUP_COMPLETE

        # Handle confirmation
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
        """Reset the setup process"""
        self.current_step = SetupStep.DETECTING_PERSON
        self.setup_confirmed = False
        self.confirmation_frames = 0
        self.smoothed_angle = None
        self.angle_history.clear()
        self.position_history.clear()


def main():
    guide = CameraPositionGuide()
    camera_manager = CameraManager()

    # Notification state
    notification_text = ""
    notification_expire_time = 0

    def on_camera_change(camera_info: CameraInfo, event: str):
        """Callback for camera connect/disconnect events"""
        nonlocal notification_text, notification_expire_time
        if event == "connected":
            notification_text = f"Camera connected: {camera_info.name}"
            print(f"\n[AUTO-DETECT] New camera connected: {camera_info.name}")
            # Auto-switch to external camera
            if camera_info.is_external:
                camera_manager.select_camera(camera_info.index)
                notification_text = f"Switched to: {camera_info.name}"
                print(f"[AUTO-SWITCH] Switched to external camera: {camera_info.name}")
        else:
            notification_text = f"Camera disconnected: {camera_info.name}"
            print(f"\n[AUTO-DETECT] Camera disconnected: {camera_info.name}")
            # If current camera disconnected, switch to another
            if camera_info.index == camera_manager.current_camera_index:
                camera_manager.detect_cameras()
                if camera_manager.select_best_camera():
                    new_cam = camera_manager.get_current_camera_info()
                    notification_text = f"Switched to: {new_cam.name if new_cam else 'Unknown'}"
        notification_expire_time = time.time() + 3.0  # Show for 3 seconds

    print("=" * 60)
    print("CAMERA POSITION GUIDE FOR AI PROCTORING")
    print("=" * 60)

    # Detect available cameras
    print("\nScanning for cameras...")
    cameras = camera_manager.detect_cameras()

    if not cameras:
        print("Error: No cameras found!")
        return

    print(f"\nFound {len(cameras)} camera(s):")
    for idx, cam in cameras.items():
        ext_tag = " [EXTERNAL]" if cam.is_external else ""
        print(f"  [{idx}] {cam.name} ({cam.width}x{cam.height}){ext_tag}")

    # Auto-select best camera (prefers external)
    if camera_manager.select_best_camera():
        current = camera_manager.get_current_camera_info()
        print(f"\nSelected: {current.name if current else 'Unknown'}")
    else:
        print("Error: Could not open any camera")
        return

    # Start auto-detection for hot-plugging
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
            # Camera might have disconnected
            print("Warning: Failed to read frame, retrying...")
            time.sleep(0.5)
            # Try to recover
            camera_manager.detect_cameras()
            if camera_manager.select_best_camera():
                continue
            else:
                print("Error: No cameras available")
                break

        # Process frame for position guidance
        processed_frame, is_complete = guide.process_frame(frame)

        # Add camera info overlay
        current_cam = camera_manager.get_current_camera_info()
        current_notification = notification_text if time.time() < notification_expire_time else ""
        processed_frame = guide.draw_camera_info(processed_frame, current_cam, current_notification)

        # Display
        cv2.imshow('Camera Position Guide - AI Proctoring', processed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            guide.reset()
            print("Setup reset - starting over...")
        elif key == ord('c') or key == ord('C'):
            # Switch camera
            new_cam = camera_manager.switch_to_next_camera()
            if new_cam:
                print(f"Switched to: {new_cam.name}")
                notification_text = f"Switched to: {new_cam.name}"
                notification_expire_time = time.time() + 2.0
                guide.reset()  # Reset setup for new camera view
            else:
                print("No other cameras available")

    # Cleanup
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
