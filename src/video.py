from pathlib import Path
import cv2


class VideoReader:

    def __init__(self, video_path):
        """Open the video file and prepare it for reading."""
        # Convert the path to a Path object
        self.video_path = Path(video_path)

        # Check whether the video file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Open the video
        self.capture = cv2.VideoCapture(str(self.video_path))

        # Check whether OpenCV successfully opened the video
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        print("Video opened successfully")

    def read_frame(self):
        """Read one frame from the video.

        Returns:
            success: True if a frame was read successfully.
            frame: The video frame, or None if the video ended.
        """
        success, frame = self.capture.read()
        return success, frame

    def release(self):
        """Release the video resource."""
        self.capture.release()


if __name__ == "__main__":
    # Find the root directory of the project
    project_directory = Path(__file__).resolve().parent.parent

    # Create the complete path to the video
    video_path = project_directory / "videos" / "test.mp4"

    # Create the video reader
    video = VideoReader(video_path)

    while True:
        # Read one frame
        success, frame = video.read_frame()

        # Stop when the video ends
        if not success:
            print("Video ended")
            break

        # Show the current frame
        cv2.imshow("Tennis Video", frame)

        # Press q to close the video
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release resources
    video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()