import pyautogui as pgui
import cv2
import time
import numpy as np
from directkeys import PressKey, ReleaseKey, W, A, S, D

def button_mash(key):
     PressKey(key)
     timeout = time.time() + 0.16
     while(time.time() < timeout):{}
     ReleaseKey(key)
     # timeout = time.time() + 0.16
     # while(time.time() < timeout):
     #     pgui.typewrite(key)


# key_list = ['A', 'D', 'W', 'S']
key_list = [A, D, W, S]

print('count down')
for i in range(4):
    print(i)
    time.sleep(1)

while(True):
    theta = np.random.randn(4)
    key = key_list[np.argmax(theta)]
    button_mash(key)
    # time.sleep(0.16)
    print('pressed ' + str(key))