# Just for testing allignment, doesnt affect the main program

import cv2
import numpy as np
from grabscreen import grab_screen
import time
from math import floor
import pyautogui
from collections import deque
from directkeys import PressKey, ReleaseKey, F1

idle_sequence = 60
min_score_sum = 10

def right_movement(image_new, image):
    diff = (np.sum(np.array(image_new) - np.array(image)) / np.sum(image)) * 100
    if diff > 5:
        return diff
    else:
        return 0

for i in range(4):
    print(i)
    time.sleep(1)


t = time.time()
image = grab_screen(region = (0, 60, 512, 444))
image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
# score_queue = deque([])

score = 0
score_queue = deque([])
array_sum = 0
curr_frame = 0
while(True):
    curr_frame += 1
    image_new = grab_screen(region = (0, 60, 512, 444))
    image_new = cv2.cvtColor(image_new, cv2.COLOR_BGRA2RGB)

    diff = (np.sum(np.array(image_new) - np.array(image))/ np.sum(image)) * 100

    font = cv2.FONT_HERSHEY_SIMPLEX
    t = time.time() - t
    fps = floor(1 / t)
    if curr_frame >= 200:
        score = right_movement(image_new, image)
        score_queue.append(score)
        if len(score_queue) == idle_sequence:
            temp_array = np.array(score_queue)
            array_sum = np.sum(temp_array)
            if array_sum < min_score_sum:
                PressKey(F1)
                print('pressed F1')
                time.sleep(0.01)
                ReleaseKey(F1)
                curr_frame = 0
            score_queue.popleft()
    image = image_new
    cv2.putText(image_new, str(fps), (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_new, str(floor(diff)), (100, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_new, str(curr_frame), (200, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('window', np.array(image_new))
    # cv2.imshow('CO', cv2.Canny(np.array(image_new), threshold1=200, threshold2=300))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


    t = time.time()