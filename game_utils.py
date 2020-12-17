from constants import stronger_gesture, computer_gestures
from cv2 import imread
import random


class Gesture:
    gesture_names = ("rock", "paper", "scissors")

    def __init__(self, name):
        self.name = name
        self.player = 0

    @staticmethod
    def generate_random():
        generated_gesture = Gesture(random.choice(Gesture.gesture_names))
        generated_gesture.player = 1
        return generated_gesture

    def __repr__(self):
        return f"Gesture({self.name})"

    def __eq__(self, other):
        return self.name == other.name

    def __gt__(self, other):
        return stronger_gesture[other.name] == self.name

    def __lt__(self, other):
        return stronger_gesture[other.name] != self.name


class RockPaperScissors:
    ids = {
        0: "person",
        1: "computer",
        2: "draw"
    }

    @classmethod
    def get_result(cls, person_gesture, computer_gesture):
        arg_list = [person_gesture, computer_gesture]
        if not person_gesture == computer_gesture:
            winner = arg_list.index(max(arg_list))
            return winner, f"{cls.ids[winner]} wins"
        else:
            return 2, "draw"

