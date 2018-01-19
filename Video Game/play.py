import pyautogui
import threading
import numpy as np
import time

for i in range(4):
    print(i)
    time.sleep(1)

#key_list = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'ctrlleft', 'shiftleft']
key_list = ['w', 'a', 's', 'd', 'leftclick', 'rightclick']
theta = np.random.randn(6, 50)
#file = open("test.txt", 'w')
#file.write(str(theta))sa
#file.close()

class pressKey(threading.Thread):
    def run(self, i):
        timeout = time.time() + 0.5
        if key_list[i] not in ['leftclick' 'rightclick']:
            while time.time() < timeout:
                pyautogui.keyDown(key_list[i])

            pyautogui.typewrite(key_list[i])
        else:
            pyautogui.dragRel(0, 0, duration=0.1)


theta = (theta > 0) * 1

t = time.time()

for i in range(50):
    presses = [pressKey() for _ in range(np.count_nonzero(theta[:, i]))]
    j = np.argwhere(theta[:, i] == 1).reshape(-1, 1)
    for k in range(np.count_nonzero(theta[:, i])):
        presses[k].run(int(j[k]))

t = time.time() - t
print(t)