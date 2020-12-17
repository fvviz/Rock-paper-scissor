x, y, w, h = 100, 100, 170, 170

model_path = "model/model2/model2.json"
model_weights_path = "model/model2/best_weights.h5"

rectangle_color = (0, 0, 255)
text_color = (0, 255, 0)

computer_gestures = {
    "rock":     "bot_gestures/rock.png",
    "paper":    "bot_gestures/paper.png",
    "scissors": "bot_gestures/scissors.png"
}

a = {
    "rock"  :  "paper",
    "paper" : "scissor",
    "scissor" : "rock"
}