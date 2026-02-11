import cv2
import mediapipe as mp
import numpy as np
import math

class PoseOrientationTracker:
    def __init__(self, smoothing_factor=0.2):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
            smooth_landmarks=True
        )

        # Base angle (reference direction in degrees)
        # 0 = facing camera, positive = turned right, negative = turned left
        self.base_body_angle = 0.0
        self.base_head_angle = 0.0

        # Threshold angle (30 degrees)
        self.threshold_angle = 30.0

        # Smoothing parameters
        self.smoothing_factor = smoothing_factor
        self.smoothed_body_angle = None
        self.smoothed_head_angle = None

    def set_base_angle(self, body_angle, head_angle):
        """Set the base reference angle"""
        self.base_body_angle = body_angle
        self.base_head_angle = head_angle
        print(f"Base angles set - Body: {body_angle:.1f}°, Head: {head_angle:.1f}°")

    def smooth_angle(self, new_angle, smoothed_angle):
        """Apply exponential moving average smoothing to an angle"""
        if smoothed_angle is None:
            return new_angle
        else:
            return self.smoothing_factor * new_angle + (1 - self.smoothing_factor) * smoothed_angle

    def calculate_body_orientation(self, landmarks, img_w, img_h):
        """Calculate body orientation from shoulder positions"""
        # Get shoulder keypoints
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Get hip keypoints for additional reference
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Convert to pixel coordinates
        left_shoulder_x = left_shoulder.x * img_w
        right_shoulder_x = right_shoulder.x * img_w
        left_hip_x = left_hip.x * img_w
        right_hip_x = right_hip.x * img_w

        # Calculate shoulder center
        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2 * img_h
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2

        # Calculate hip center
        hip_center_x = (left_hip_x + right_hip_x) / 2
        hip_center_z = (left_hip.z + right_hip.z) / 2

        # Shoulder width (to determine if turned)
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)

        # Average shoulder width when facing camera (normalized)
        # When turned sideways, shoulder width appears smaller
        facing_ratio = shoulder_width / img_w

        # Calculate orientation angle from shoulder positions
        # Use z-depth difference between shoulders
        # When turned right, left shoulder is farther (larger z), right is closer (smaller z)
        z_diff = left_shoulder.z - right_shoulder.z

        # Calculate angle based on shoulder relative positions and z-depth
        # Positive angle = turned to the right, negative = turned to the left
        body_angle = math.degrees(math.atan2(z_diff, facing_ratio))

        # Adjust based on torso twist (shoulder vs hip alignment)
        torso_twist = (shoulder_center_x - hip_center_x) / img_w * 100
        body_angle += torso_twist

        return body_angle, shoulder_center_x, shoulder_center_y

    def calculate_head_orientation(self, landmarks, img_w, img_h):
        """Calculate head orientation from nose and ear positions"""
        # Get head keypoints
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]

        # Convert to pixel coordinates
        nose_x = nose.x * img_w
        nose_y = nose.y * img_h
        left_ear_x = left_ear.x * img_w
        right_ear_x = right_ear.x * img_w
        left_eye_x = left_eye.x * img_w
        right_eye_x = right_eye.x * img_w

        # Calculate head center
        head_center_x = (left_ear_x + right_ear_x) / 2
        head_center_y = (left_ear.y + right_ear.y) / 2 * img_h

        # Ear width (indicates head rotation)
        ear_width = abs(right_ear_x - left_ear_x)
        eye_width = abs(right_eye_x - left_eye_x)

        # Normalized width ratio
        ear_ratio = ear_width / img_w
        eye_ratio = eye_width / img_w

        # Calculate head angle using z-depth
        # When head turns right, left ear is farther, right ear is closer
        z_diff = left_ear.z - right_ear.z

        # Head angle calculation
        head_angle = math.degrees(math.atan2(z_diff, ear_ratio + eye_ratio))

        # Adjust based on nose position relative to ear center
        nose_offset = (nose_x - head_center_x) / img_w * 100
        head_angle += nose_offset

        return head_angle, nose_x, nose_y

    def is_within_threshold(self, current_angle, base_angle):
        """Check if current angle is within threshold of base angle"""
        angle_diff = abs(current_angle - base_angle)
        return angle_diff <= self.threshold_angle, angle_diff

    def draw_orientation(self, img, body_angle, head_angle, body_center, head_pos):
        """Draw orientation vectors"""
        arrow_length = 150

        # Draw body orientation arrow (blue)
        body_rad = math.radians(body_angle)
        body_end_x = int(body_center[0] + arrow_length * math.sin(body_rad))
        body_end_y = int(body_center[1] - arrow_length * math.cos(body_rad) * 0.3)
        cv2.arrowedLine(img,
                       (int(body_center[0]), int(body_center[1])),
                       (body_end_x, body_end_y),
                       (255, 150, 0), 4, tipLength=0.3)
        cv2.putText(img, "BODY", (body_end_x + 10, body_end_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # Draw head orientation arrow (green)
        head_rad = math.radians(head_angle)
        head_end_x = int(head_pos[0] + arrow_length * math.sin(head_rad))
        head_end_y = int(head_pos[1] - arrow_length * math.cos(head_rad) * 0.3)
        cv2.arrowedLine(img,
                       (int(head_pos[0]), int(head_pos[1])),
                       (head_end_x, head_end_y),
                       (0, 255, 150), 4, tipLength=0.3)
        cv2.putText(img, "HEAD", (head_end_x + 10, head_end_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)

    def process_frame(self, frame):
        """Process a single frame"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        img_h, img_w = frame.shape[:2]

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Calculate orientations
            body_angle, body_x, body_y = self.calculate_body_orientation(landmarks, img_w, img_h)
            head_angle, head_x, head_y = self.calculate_head_orientation(landmarks, img_w, img_h)

            # Apply smoothing
            self.smoothed_body_angle = self.smooth_angle(body_angle, self.smoothed_body_angle)
            self.smoothed_head_angle = self.smooth_angle(head_angle, self.smoothed_head_angle)

            # Check if within threshold
            body_within, body_diff = self.is_within_threshold(self.smoothed_body_angle, self.base_body_angle)
            head_within, head_diff = self.is_within_threshold(self.smoothed_head_angle, self.base_head_angle)

            # Draw orientation arrows
            self.draw_orientation(frame, self.smoothed_body_angle, self.smoothed_head_angle,
                                (body_x, body_y), (head_x, head_y))

            # Display info
            y_offset = 30

            # # Body angle info
            # cv2.putText(frame, f"Body Angle: {self.smoothed_body_angle:.1f} deg",
            #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
            # y_offset += 30
            # cv2.putText(frame, f"Base Body: {self.base_body_angle:.1f} deg",
            #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 120, 0), 2)
            # y_offset += 30

            body_color = (0, 255, 0) if body_within else (0, 0, 255)
            body_status = "WITHIN 30°" if body_within else f"OUT ({body_diff:.1f}°)"
            cv2.putText(frame, f"Body: {body_status}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, body_color, 2)
            y_offset += 40

            # # Head angle info
            # cv2.putText(frame, f"Head Angle: {self.smoothed_head_angle:.1f} deg",
            #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
            # y_offset += 30
            # cv2.putText(frame, f"Base Head: {self.base_head_angle:.1f} deg",
            #            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 120), 2)
            # y_offset += 30

            head_color = (0, 255, 0) if head_within else (0, 0, 255)
            head_status = "WITHIN 30°" if head_within else f"OUT ({head_diff:.1f}°)"
            cv2.putText(frame, f"Head: {head_status}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)
            y_offset += 30

            # Smoothing info
            cv2.putText(frame, f"Smoothing: {self.smoothing_factor:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

            # Instructions
            cv2.putText(frame, "S: Set base | R: Reset | +/-: Smoothing",
                       (10, img_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return frame, self.smoothed_body_angle, self.smoothed_head_angle

        # No pose detected
        cv2.putText(frame, "No pose detected - Stand in frame", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame, None, None


def main():
    # Initialize tracker
    tracker = PoseOrientationTracker(smoothing_factor=0.2)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Pose Orientation Tracking Started!")
    print("Controls:")
    print("  'S' - Set current orientation as base reference")
    print("  'R' - Reset base reference to (0, 0)")
    print("  '+' - Increase smoothing (more responsive)")
    print("  '-' - Decrease smoothing (more stable)")
    print("  'Q' - Quit")

    last_body_angle = None
    last_head_angle = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        # Process frame
        processed_frame, body_angle, head_angle = tracker.process_frame(frame)

        # Store last valid angles
        if body_angle is not None:
            last_body_angle = body_angle
            last_head_angle = head_angle

        # Display frame
        cv2.imshow('Pose Orientation Tracking - 30 Degree Angle', processed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            if last_body_angle is not None:
                tracker.set_base_angle(last_body_angle, last_head_angle)
        elif key == ord('r') or key == ord('R'):
            tracker.set_base_angle(0.0, 0.0)
        elif key == ord('+') or key == ord('='):
            tracker.smoothing_factor = min(1.0, tracker.smoothing_factor + 0.05)
            print(f"Smoothing factor: {tracker.smoothing_factor:.2f}")
        elif key == ord('-') or key == ord('_'):
            tracker.smoothing_factor = max(0.05, tracker.smoothing_factor - 0.05)
            print(f"Smoothing factor: {tracker.smoothing_factor:.2f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
