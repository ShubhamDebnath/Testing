import win32api as wapi
import time
from grabscreen import grab_screen
import os
import cv2
import numpy as np

def getKey():
    key_list = ['A', 'W', 'S', 'D', 'K']
    keys = []
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def key2output(keys):
    k_output = [0,0,0,0]
    if 'W' in keys and 'A' in keys:
        k_output = [1, 1, 0, 0]
    elif 'W' in keys and 'D' in keys:
        k_output = [0, 1, 0, 1]
    elif 'S' in keys and 'A' in keys:
        k_output = [1, 0, 1, 0]
    elif 'S' in keys and 'D' in keys:
        k_output = [0, 0, 1, 1]
    elif 'W' in keys:
        k_output = [0, 1, 0, 0]
    elif 'S' in keys:
        k_output = [0, 0, 1, 0]
    elif 'A' in keys:
        k_output = [1, 0, 0, 0]
    elif 'D' in keys:
        k_output = [0, 0, 0, 1]
    else:
        k_output = [0, 0, 0, 0]

    return k_output


file_name = "C:\\Users\\gamef\\Desktop\\training_data.npy"

if os.path.isfile(file_name):
    print("file exists")
    training_data = list(np.load(file_name))
else:
    print("creating new file")
    training_data = []

for i in range(4):
    print(i)
    time.sleep(1)

paused = False
while(True):
    if not paused:
        screen = grab_screen(region=(0, 40, 1024, 808))
        screen = cv2.resize(screen, (400, 225))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        keys = getKey()
        output = key2output(keys)
        training_data.append([screen, output])

        if len(training_data) % 100 == 0:
            print(len(training_data))

        if len(training_data) % 500 == 0:
            np.save(file_name, training_data)
            print('Saved')

    keys = getKey()
    if 'K' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)