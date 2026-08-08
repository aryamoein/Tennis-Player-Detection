import threading
import time
from pathlib import Path

import cv2


class ThreadedCapture:
    """
    Capture frames in a background thread and always
    expose the most recent frame.

    Advantages on a weak board:

    - Camera capture never blocks inference.
    - Stale frames are dropped instead of queued.
    - A single reusable frame buffer avoids copies.
    """

    def __init__(
        self,
        source,
        width=None,
        height=None,
        fps=None,
        use_mjpeg=True,
        buffersize=1,
    ):
        """
        Args:
            source:
                Camera index (int) or path to a video file.
            width, height, fps:
                Optional capture properties (camera only).
            use_mjpeg:
                Request MJPG format from the camera.
                Much cheaper to decode on weak CPUs.
            buffersize:
                Internal frame queue size (keep at 1 for low latency).
        """

        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._use_mjpeg = use_mjpeg
        self._buffersize = buffersize

        self._capture = None
        self._frame = None
        self._finished = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None


    def start(self):
        """Open the source and start the capture thread."""

        if isinstance(self._source, (str, Path)):

            self._capture = cv2.VideoCapture(
                str(self._source)
            )

        else:

            self._capture = cv2.VideoCapture(
                int(self._source)
            )


        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: {self._source}"
            )


        if isinstance(self._source, int):

            if self._use_mjpeg:

                self._capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG")
                )

            if self._width:
                self._capture.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    int(self._width)
                )

            if self._height:
                self._capture.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    int(self._height)
                )

            if self._fps:
                self._capture.set(
                    cv2.CAP_PROP_FPS,
                    float(self._fps)
                )

            self._capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                int(self._buffersize)
            )

            self._file_delay = None

        else:

            # Throttle video playback to its native speed so
            # the whole file does not finish instantly.

            playback_fps = (
                self._capture.get(cv2.CAP_PROP_FPS)
            )

            if playback_fps and playback_fps > 0:
                self._file_delay = 1.0 / playback_fps
            else:
                self._file_delay = None


        self._running = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self._thread.start()


    def _loop(self):
        """Background capture loop."""

        while self._running:

            success, frame = self._capture.read()

            if not success:

                # A video has ended; stop the thread.
                # A camera had a hiccup; keep trying.

                if isinstance(self._source, (str, Path)):
                    self._finished = True
                    break

                continue


            with self._lock:
                self._frame = frame


            if self._file_delay:
                time.sleep(self._file_delay)


    def read(self):
        """
        Return the most recent frame.

        Returns:
            Latest frame, or None if no frame is ready yet
            (or the video file has ended).
        """

        if self._finished:
            return None

        with self._lock:
            frame = self._frame

        return frame


    def is_file(self):
        return isinstance(
            self._source,
            (str, Path)
        )

    def finished(self):
        """True when a video file has reached the end."""

        return self._finished


    def stop(self):
        """Stop the capture thread and release the source."""

        self._running = False

        if self._thread:
            self._thread.join(timeout=1)


        if self._capture:
            self._capture.release()