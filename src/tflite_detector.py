from pathlib import Path

import cv2
import numpy as np

# Import TensorFlow Lite lazily so that code importing this
# module still works where TFLite is not installed.

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        Interpreter = None


def letterbox(image, size, fill_value=114):
    """
    Resize an image to a square of `size` keeping the aspect ratio.

    Returns:
        (canvas, scale, left, top)
        scale: applied resize factor
        left, top: pad offsets in the letterboxed image
    """

    height, width = image.shape[:2]

    scale = min(
        size / width,
        size / height
    )

    new_width = max(
        int(round(width * scale)),
        1
    )

    new_height = max(
        int(round(height * scale)),
        1
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas = np.full(
        (size, size, 3),
        fill_value=fill_value,
        dtype=image.dtype
    )

    canvas[
        top:top + new_height,
        left:left + new_width
    ] = resized

    return canvas, scale, left, top


class TFLiteDetector:
    """
    Person detection using a TensorFlow Lite model.

    Supports YOLO-style heads (single raw output tensor)
    and post-processed heads with built-in NMS
    (boxes, classes, scores, num_detections).

    Provides the same detect_person() interface as YOLODetector.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_path, threads=4):
        if Interpreter is None:
            raise ImportError(
                "TensorFlow Lite is not installed. "
                "Install 'tflite-runtime' on the Raspberry Pi."
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self._interpreter = Interpreter(
            model_path=str(self.model_path),
            num_threads=int(threads)
        )

        self._interpreter.allocate_tensors()

        self._input_details = (
            self._interpreter.get_input_details()
        )

        self._output_details = (
            self._interpreter.get_output_details()
        )

        input_detail = self._input_details[0]

        self._input_index = input_detail["index"]

        self._input_shape = (
            input_detail["shape"]
        )

        # Fixed square inference size
        # [1, size, size, 3] or [1, 3, size, size]

        self._inference_size = int(self._input_shape[1])

        self._input_dtype = np.dtype(
            input_detail["dtype"]
        )

        if self._input_dtype == np.uint8:

            # Quantized input: feed pixels 0..255 as-is.
            self._scale_bytes = True

        else:

            # Float input: divide by 255.
            self._scale_bytes = False

        self._prepare_output_details()

        print(
            f"TFLite model loaded: {self.model_path.name}"
        )

        print(
            f"Input shape: {self._input_shape}"
        )


    def _prepare_output_details(self):
        """
        Identify the detection output layout.

        Two layouts are supported:

        1. YOLO-style raw head (single tensor [1, 84, N]).
        2. Post-processed TFLite detection head (built-in NMS).
           The standard published order is:

               output 0: detection_boxes    (1, N, 4)
               output 1: detection_classes  (1, N)
               output 2: detection_scores   (1, N)
               output 3: num_detections     (1)
        """

        self._yolo_head = False
        self._yolo_index = None

        self._boxes_index = 0
        self._classes_index = 1
        self._scores_index = 2
        self._num_index = None

        num_outputs = len(self._output_details)

        if num_outputs > 3:
            self._num_index = 3

        for index, detail in enumerate(
            self._output_details
        ):

            shape = tuple(detail["shape"])

            # YOLO head: [1, 84, N].
            if (
                len(shape) == 3
                and shape[1] == 84
            ):
                self._yolo_head = True
                self._yolo_index = index
                return

            # Post-processed head: boxes end with a 4.
            if (
                len(shape) == 3
                and shape[2] == 4
            ):
                self._boxes_index = index


    def detect_person(
        self,
        frame,
        confidence_threshold=0.5
    ):
        """
        Detect the most confident person.

        Returns:
            Same dictionary format as YOLODetector:
            class_name, confidence, x1, y1, x2, y2

            or None
        """

        blob, scale, left, top = (
            self._preprocess(frame)
        )

        outputs = self._run(blob)

        if self._yolo_head:

            return self._parse_yolo(
                outputs,
                scale, left, top,
                confidence_threshold
            )

        return self._parse_postprocessed(
            outputs,
            scale, left, top,
            confidence_threshold
        )


    def _preprocess(self, frame):
        """Convert frame to the model input tensor."""

        # Models are trained on RGB images.

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        blob, scale, left, top = letterbox(
            rgb,
            self._inference_size
        )

        # Add batch dimension -> (1, size, size, 3)

        blob = np.expand_dims(
            blob,
            axis=0
        )

        if not self._scale_bytes:

            blob = blob.astype(np.float32) / 255.0

        return blob, scale, left, top


    def _run(self, blob):
        """Run inference and return all outputs."""

        self._interpreter.set_tensor(
            self._input_index,
            blob
        )

        self._interpreter.invoke()

        return [
            self._interpreter.get_tensor(
                detail["index"]
            )
            for detail in self._output_details
        ]


    def _map_box(self, x1, y1, x2, y2, scale, left, top):
        """
        Map box coordinates from model input to original frame.
        """

        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale

        return int(x1), int(y1), int(x2), int(y2)


    def _parse_yolo(
        self,
        outputs,
        scale, left, top,
        confidence_threshold
    ):
        """
        Parse a YOLO-style head (single tensor [1, 84, N]).
        """

        predictions = outputs[self._yolo_index][0]

        predictions = predictions.T

        best_detection = None

        best_confidence = confidence_threshold


        for prediction in predictions:

            x_center = prediction[0]
            y_center = prediction[1]

            width = prediction[2]
            height = prediction[3]

            confidence = prediction[
                4 + self.PERSON_CLASS_ID
            ]

            if confidence < best_confidence:
                continue

            best_confidence = float(confidence)

            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2

            x1, y1, x2, y2 = self._map_box(
                x1, y1, x2, y2,
                scale, left, top
            )

            best_detection = {
                "class_name": "person",
                "confidence": best_confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }

        return best_detection


    def _parse_postprocessed(
        self,
        outputs,
        scale, left, top,
        confidence_threshold
    ):
        """
        Parse a post-processed head with built-in NMS
        (SSD / EfficientDet style).
        """

        boxes = outputs[self._boxes_index][0]

        scores = outputs[self._scores_index][0]

        classes = outputs[self._classes_index][0]

        frame_size = self._inference_size

        num_detections = len(scores)

        if self._num_index is not None:

            detected = int(outputs[self._num_index][0])

            if detected > 0:
                num_detections = min(detected, num_detections)

        # The built-in NMS returns the strongest detections first,
        # so class 0 (person) within the top detections wins.
        # Scores below the threshold are ignored.

        for i in range(num_detections):

            class_id = int(classes[i])

            confidence = float(scores[i])

            if class_id != self.PERSON_CLASS_ID:
                continue

            if confidence < confidence_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            x1, y1, x2, y2 = self._map_box(
                xmin * frame_size,
                ymin * frame_size,
                xmax * frame_size,
                ymax * frame_size,
                scale, left, top
            )

            return {
                "class_name": "person",
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }

        return None