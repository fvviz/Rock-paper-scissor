from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
import numpy as np
import time
import random

from constants import x, y, w, h
from constants import model_path, model_weights_path
from constants import rectangle_color, text_color


class GestureModel:
    def __init__(self, model_path_, model_weights_path_):
        self.model = model_from_json(open(model_path_,"r").read())
        self.model.load_weights(model_weights_path_)
        self.gestures = ('empty', 'paper', 'rock', 'scissors')

    @classmethod
    def preprocess(cls, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray_frame[y:y + w, x:x + h]
        roi = cv2.resize(roi, (50, 50))
        img_pixels = img_to_array(roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        return img_pixels

    def predict(self, frame):
        img_pixels = self.preprocess(frame)
        prediction = self.model.predict(img_pixels)
        max_index = np.argmax(prediction[0])

        predicted_gesture = self.gestures[max_index]
        predict_percent = prediction[0][max_index]*100
        return predicted_gesture, round(predict_percent, 2)


class WebCam:

    model = GestureModel(model_path,
                         model_weights_path)

    @classmethod
    def create_rectangle(cls, frame):
        cv2.rectangle(img=frame,
                      pt1=(x, y), pt2=(x+w, y+h),
                      color=rectangle_color, thickness=5)

    @classmethod
    def create_text(cls, frame, text, font_scale=1, thickness=2, org=(x+300, y), color = text_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img=frame, text=text,
                    org=org,
                    fontFace=font, fontScale=font_scale,
                    color=color, thickness=thickness)

    @classmethod
    def add_prediction(cls, frame):
        gesture, percent = cls.model.predict(frame)
        cls.create_rectangle(frame)

        flipped_frame = cv2.flip(frame, 1)
        cls.create_text(flipped_frame, f"{gesture} {percent}%")

    @classmethod
    def play_game(cls):
        cap = cv2.VideoCapture(0)
        frame_num = 0
        val = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            print("val" , val)

            if val < 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cls.create_rectangle(gray_frame)

                frame = cv2.flip(gray_frame, 1)

                cls.create_text(frame,
                                f"{val}", org=(x + 300, y + 150),
                                font_scale=5, thickness=5)

                cls.create_text(frame, "get ready",
                                org=(x + 300, y + 200),
                                font_scale=1, thickness=2)
                frame = cv2.resize(frame, (1000, 700))

                if frame_num % 50 == 0:
                    val += 1
                frame_num += 1
            else:
                gesture, percent = cls.model.predict(frame)
                cls.create_rectangle(frame)
                frame = cv2.flip(frame, 1)

                cls.create_text(frame, f"{gesture} {percent}%")
                frame = cv2.resize(frame, (1000, 700))
            cv2.imshow('Rock Paper Scissors!', frame)
            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                    break




    @classmethod
    def start(cls):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            flipped_frame = cv2.flip(frame, 1)
            resized_frame = cv2.resize(flipped_frame, (1000, 700))
            cv2.imshow('Rock Paper Scissors!', resized_frame)
            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                break

    @classmethod
    def start_predictions(cls):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            preparing = True
            gesture, percent = cls.model.predict(frame)
            cls.create_rectangle(frame)

            flipped_frame = cv2.flip(frame, 1)
            cls.create_text(flipped_frame, f"{gesture} {percent}%")
            resized_frame = cv2.resize(flipped_frame, (1000, 700))
            cv2.imshow('Rock Paper Scissors!', resized_frame)
            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                break

        cap.release()
        cv2.destroyWindow()






























