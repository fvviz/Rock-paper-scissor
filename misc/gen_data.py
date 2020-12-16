import cv2
import os
from PIL import Image
import time

cap = cv2.VideoCapture(0)
start = False
data_type = "validation"
current_class = "scissors"
dir = f"data/{data_type}/{current_class}"
x, y, w, h =  200, 100, 170, 170
total_samples = 500
available_samples = len(os.listdir(dir))

while not start:
    ret, test_img = cap.read()
    if not ret:
        continue
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(test_img,
                f"press s to start",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1)
    cv2.putText(test_img,
                f"class:{current_class}",
                (x, y + h + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1)
    cv2.putText(test_img,
                f"type:{data_type}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1)
    cv2.putText(test_img,
                f"samples available:{available_samples}/{total_samples}",
                (x, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('s'):
        start = True

    if cv2.waitKey(10) == ord('q'):
        break



for i in range(available_samples, total_samples):
        time.sleep(0.07)
        print(i)
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image

        if not ret:
            continue

        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(test_img,
                    f"Capturing {current_class}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(test_img,
                    f"class:{current_class}",
                    (x, y + h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 0), 1)
        cv2.putText(test_img,
                    f"type:{data_type}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 0), 1)
        cv2.putText(test_img,
                    f"samples captured:{i}/{total_samples}",
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 0), 1)

        gray_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        print(gray_image)

        roi = gray_image[y:y+h, x:x+w]
        roi_image = Image.fromarray(roi, "L")
        roi_image_resized= roi_image.resize((50,50))
        roi_image_resized.save(f"{dir}/{i}.png")
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Emotion analysis', resized_img)
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

cap.release()
cv2.destroyWindow()
