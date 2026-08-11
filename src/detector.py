from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:

    def __init__(self, model_path, inference_size=640, threads=1):
        """
        Load the YOLO ONNX model.

        Args:
            model_path:
                Path to the ONNX model file.

            inference_size:
                Square input size the model was exported at
                (e.g. 640 or 320). Used to resize the frame
                and map detections back to the original size.

            threads:
                ONNX runtime CPU threads (e.g. 4 on the
                Pi Zero 2 W). A big speedup for convolutions.
        """

        self.model_path = Path(model_path)

        self.inference_size = inference_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Load ONNX model with the CPU thread pool tuned for
        # the board. The default single thread heavily wastes
        # a quad-core Pi.

        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = int(threads)

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )

        print("YOLO model loaded successfully")


        # Get input information
        model_input = self.session.get_inputs()[0]

        self.input_name = model_input.name
        self.input_shape = model_input.shape

        print("Input name:", self.input_name)
        print("Input shape:", self.input_shape)


        # Get output information
        outputs = self.session.get_outputs()

        print("Number of outputs:", len(outputs))

        for output in outputs:
            print(
                "Output:",
                output.name,
                output.shape
            )


    def predict(self, frame):
        """
        Run YOLO inference on one frame.
        """

        # Resize frame to YOLO input size
        resized_frame = cv2.resize(
            frame,
            (self.inference_size, self.inference_size)
        )


        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            resized_frame,
            cv2.COLOR_BGR2RGB
        )


        # Normalize pixels
        normalized_frame = (
            rgb_frame.astype(np.float32) / 255.0
        )


        # HWC -> CHW
        input_tensor = np.transpose(
            normalized_frame,
            (2, 0, 1)
        )


        # Add batch dimension
        # (3,640,640) -> (1,3,640,640)

        input_tensor = np.expand_dims(
            input_tensor,
            axis=0
        )


        # Run inference

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor
            }
        )


        return outputs[0]



    def detect_person(self, frame, confidence_threshold=0.5):
        """
        Detect the person with highest confidence.

        Returns:
            Dictionary with:
            - confidence
            - bounding box coordinates

            or None
        """


        # Run YOLO

        output = self.predict(frame)


        # Remove batch dimension
        # (1,84,8400)
        #      |
        #      v
        # (84,8400)

        predictions = output[0]


        # Change:
        # (84,8400)
        #
        # to:
        #
        # (2100,84)

        predictions = predictions.T


        person_class_id = 0


        # Person confidences for every anchor (vectorised,
        # much faster than a Python loop).

        confidences = predictions[:, 4 + person_class_id]

        best_index = int(np.argmax(confidences))

        best_confidence = float(confidences[best_index])


        if best_confidence < confidence_threshold:

            return None


        # Bounding box in YOLO format

        x_center, y_center, width, height = predictions[best_index, :4]


        # Convert center/width/height to corners, then scale
        # from model coordinates back to the original frame.

        original_height, original_width = frame.shape[:2]

        scale_x = original_width / self.inference_size
        scale_y = original_height / self.inference_size

        x1 = int((x_center - width / 2) * scale_x)
        y1 = int((y_center - height / 2) * scale_y)

        x2 = int((x_center + width / 2) * scale_x)
        y2 = int((y_center + height / 2) * scale_y)


        return {

            "class_name": "person",

            "confidence": best_confidence,

            "x1": x1,
            "y1": y1,

            "x2": x2,
            "y2": y2
        }



if __name__ == "__main__":


    project_directory = (
        Path(__file__).resolve().parent.parent
    )


    model_path = (
        project_directory
        / "models"
        / "yolov8n.onnx"
    )


    detector = YOLODetector(model_path)