from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:

    def __init__(self, model_path):
        """
        Load the YOLO ONNX model.
        """

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
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
            (640, 640)
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
        # (8400,84)

        predictions = predictions.T



        person_class_id = 0


        best_detection = None

        best_confidence = confidence_threshold



        for prediction in predictions:


            # Bounding box in YOLO format

            x_center = prediction[0]
            y_center = prediction[1]

            width = prediction[2]
            height = prediction[3]


            # Person confidence

            confidence = prediction[
                4 + person_class_id
            ]


            if confidence < best_confidence:
                continue



            best_confidence = confidence



            # Convert:
            # center format
            #
            # x_center,y_center,w,h
            #
            # to:
            #
            # x1,y1,x2,y2


            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2



            # -------------------------------
            # SCALE TO ORIGINAL FRAME SIZE
            # -------------------------------


            original_height, original_width = frame.shape[:2]


            scale_x = original_width / 640
            scale_y = original_height / 640



            x1 *= scale_x
            x2 *= scale_x

            y1 *= scale_y
            y2 *= scale_y



            best_detection = {

                "class_name": "person",

                "confidence": float(confidence),

                "x1": int(x1),
                "y1": int(y1),

                "x2": int(x2),
                "y2": int(y2)
            }



        return best_detection



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