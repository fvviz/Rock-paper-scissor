from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
import numpy as np

from constants import roi_value, roi_location
from constants import model_path, model_weights_path


class GestureModel:
    def __init__(self,model_path, model_weights_path):
        self.model = model_from_json(open(model_path,"r").read())
        self.model.load_weights(model_weights_path)
        self.gestures = ('empty', 'paper', 'rock', 'scissors')

    @classmethod
    def preprocess(cls, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray_frame[roi_value[1]:.roi_value[1]+roi_value[2], roi_value[0]:cls.roi_value[0]+cls.roi_value[3]]
        roi = cv2.resize(roi, (50, 50))
        img_pixels = img_to_array(roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        return img_pixels

    def predict(self, frame):
        img_pixels = self.preprocess(frame)
        prediction = self.model.predict(img_pixels)
        max_index = np.argmax(prediction[0])

        predicted_gesture = self.gestures[max_index]
        predict_percent= prediction[0][max_index]*100
        return predicted_gesture, round(predict_percent, 2)


class WebCam:

    model = GestureModel(model_path,
                         model_weights_path)

    @classmethod
    def create_rectangle(cls, frame):
        rgb_tuple = (0, 0, 225)
        cv2.rectangle(img=frame,
                      pt1=roi_location[0], pt2=roi_location[1],
                      color=rgb_tuple, thickness=5)

    @classmethod
    def create_text(cls, frame, text):
        rgb_tuple = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img=frame, text=text,
                    org=(roi_location[0][0], roi_location[0][1]),
                    fontFace=font, fontScale=1,
                    color=rgb_tuple, thickness=2)

    @classmethod
    def start(cls):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            flipped_frame = cv2.flip(frame, 1)
            gesture, percent = cls.model.predict(frame)

            cls.create_rectangle(flipped_frame)
            cls.create_text(flipped_frame, f"{gesture} {percent}")

            resized_frame = cv2.resize(flipped_frame, (1000, 700))
            cv2.imshow('Rock Paper Scissors!', resized_frame)
            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                break

        cap.release()
        cv2.destroyWindow()












