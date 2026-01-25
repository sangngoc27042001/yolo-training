import cv2
from pathlib import Path

# Configuration variables
VIDEO_PATH = "src/pose_object_detection_running/videos/Muhammad Zohaib - AI Engineer.mp4"
OUTPUT_FOLDER = "output_frames"
INTERVAL = 15  # seconds


def capture_frames(video_path, output_folder, interval):
    """
    Capture frames from a video at specified intervals.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder for frames
        interval (float): Interval in seconds between frame captures
    """
    # Get video name without extension
    video_name = Path(video_path).stem

    # Create output folder with video name as subfolder
    output_path = Path(output_folder) / video_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Capturing frames every {interval} seconds")

    # Calculate frame interval
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            output_filename = output_path / f"frame_{saved_count:05d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(str(output_filename), frame)
            print(f"Saved: {output_filename}")
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"\nCompleted! Saved {saved_count} frames to {output_path}")


if __name__ == "__main__":
    capture_frames(VIDEO_PATH, OUTPUT_FOLDER, INTERVAL)
