import pyautogui
import threading
import numpy as np
import time
import cv2
from math import floor
from grabscreen import grab_screen

for i in range(4):
    print(i)
    time.sleep(1)

key_list = ['a', 'w', 's', 'd']


class pressKey(threading.Thread):
    def run(self, i):
        timeout = time.time() + 0.5
        while time.time() < timeout:
            pyautogui.keyDown(key_list[i])

        pyautogui.typewrite(key_list[i])

t = time.time()

while True:
    #screen = np.array(ImageGrab.grab(bbox = (0, 40, 1024, 740)))
    screen = grab_screen(region=(0, 40, 1024, 808))
    #screen = cv2.resize(screen, (400, 225))
    #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    t = time.time() - t
    fps = floor(1/ t)
    cv2.putText(screen, str(fps), (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #cv2.imshow('CrossOut', screen)
    cv2.imshow('CO', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    #cv2.imshow('CO', cv2.Canny(screen, threshold1 = 200, threshold2 = 300))
    print(" took {} seconds ", t)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    t = time.time()
    #theta = np.random.randn(4)
    # theta = (theta > 0.75) * 1
    # presses = [pressKey() for _ in range(np.count_nonzero(theta))]
    # j = np.argwhere(theta == 1)
    # for k in range(np.count_nonzero(theta)):
    #     presses[k].run(int(j[k]))
