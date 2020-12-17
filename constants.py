x, y, w, h = 400, 100, 170, 170
paste_coords = (x-300, 100)

model_path = "model/model2/model2.json"
model_weights_path = "model/model2/best_weights.h5"

rectangle_color = (0, 0, 255)
text_color = (0, 255, 0)

computer_gestures = {
    "rock":     "bot_gestures/rock.png",
    "paper":    "bot_gestures/paper.png",
    "scissors": "bot_gestures/scissors.png"
}

stronger_gesture = {
    "rock":  "paper",
    "paper": "scissor",
    "scissors": "rock"
}