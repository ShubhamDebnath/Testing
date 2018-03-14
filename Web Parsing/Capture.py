import pyautogui
import time
#from win32gui import GetWindowText, GetForegroundWindow


#game = GetWindowText(GetForegroundWindow())
base = "pic_data/"
i=1

while True:
    file_handler = open(base + "data.txt", "a")
    pic = pyautogui.screenshot()
    pic.save(base + str(i) + ".png")
    file_handler.write(str(i) + "\t" + str(pyautogui.position()) + "\n")
    i=i+1
    file_handler.close()
    time.sleep(2)

