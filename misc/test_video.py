import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

x, y, w, h =  200, 100, 170, 170
model = model_from_json(open("../model/model2/model2.json", "r").read())
model.load_weights("../model/model2/best_weights.h5")



cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, test_img = cap.read()
    if not ret:
        continue
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=7)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    roi_gray = cv2.resize(roi_gray, (50, 50))
    img_pixels = img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    gesture = ('empty','paper', 'rock', 'scissors')
    predicted_gesture = gesture[max_index]
    predict_percent = predictions[0][max_index]*100

    test_img = cv2.flip(test_img, 1)
    cv2.putText(test_img,f"{predicted_gesture} {round(predict_percent,2)}%", (int(x+100), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(test_img, f"{x} {y}", (int(x + 100), int(y+200)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    resized_img = cv2.resize(test_img, (1000, 700))

    cv2.imshow('rock paper scissors', resized_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyWindow()