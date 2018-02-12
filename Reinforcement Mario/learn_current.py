import numpy as np
# import tensorflow as tf
from collections import deque
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, F1, A, W, S, D
# from minibatch_generator import *
from conv_filter import inception_v3 as googlenet
import time
import cv2
from math import floor

num_games = 10
num_gens = 100
idle_sequence = 80
min_score_sum = 10
possible_keys = [W, A, S, D]
WIDTH = 512
HEIGHT = 384
LR = 1e-3
batch_size = 10
MODEL_NAME = "Play_Mario.model"
model = googlenet(WIDTH, HEIGHT, 3, LR, 4)

def right_movement(image_new, image):
    diff = (np.sum(image_new - image) / np.sum(image)) * 100
    if diff > 5:
        return diff
    else:
        return 0


def button_mash(key, prev_key=A):
    if key == W:
        ReleaseKey(prev_key)
        timeout = time.time() + 0.02
        PressKey(key)
        while time.time() < timeout:{}
    else:
        ReleaseKey(prev_key)
        PressKey(key)

print("Starting")
for i in range(4)[::-1]:
    print(i)
    time.sleep(1)


# let's try with 100 generations
for gen in range(num_gens):

    if gen>0:
        model.load(MODEL_NAME)

    data = []
    data_len = 0
    # 50 games in one generation
    for games in range(num_games):
        print('currently playing game number {0} in generation {1}'.format(games, gen))

        PressKey(F1)
        print('pressed F1')
        time.sleep(0.001)
        ReleaseKey(F1)
        curr_frame = 0
        image = grab_screen(region=(0, 60, WIDTH-1, HEIGHT+59))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        score_queue = deque([])
        array_sum = 0
        score = 0
        t = time.time()
        prev_key = A
        current_profit_sequence = 0

        # start loop for 400 in-game seconds
        while (True):
            # this group belongs to taking a screen shot of current frame and changing color
            curr_frame += 1
            image_new = grab_screen(region=(0, 60, WIDTH-1, HEIGHT+59))
            image_new = cv2.cvtColor(image_new, cv2.COLOR_BGRA2RGB)
            # print (image_new.shape)

            # this group is for sending the image as input and pressing the key
            if gen == 0:
                theta = np.random.randn(4)
            else:
                theta = model.predict([image_new.reshape(WIDTH,HEIGHT,3)])[0]
            key = np.argmax(theta)
            result = (theta == np.amax(theta))*1
            key_to_press = possible_keys[key]
            button_mash(key_to_press, prev_key)
            score = right_movement(image_new, image)
            # print('pressed ' + str(result))

            # creating a list of all possible the in-game frames which is input for CNN
            # if curr_frame not in img_index:
            #     img_index.append(curr_frame)
            # NOTE: list of same images created again and again for each game
            current_profit_sequence = score + current_profit_sequence*0.5
            if current_profit_sequence < 0:
                current_profit_sequence = 0
            if current_profit_sequence > 5:
                data.append([image_new, result])
                data_len +=1

            if curr_frame >= 50:
                score = right_movement(image_new, image)
                score_queue.append(score)
                if len(score_queue) == idle_sequence:
                    temp_array = np.array(score_queue)
                    array_sum = np.sum(temp_array)
                    if array_sum < min_score_sum:
                        # data = data[:-20 or None]
                        # data_len = (data_len-20) if data_len-20>0 else 0
                        break
                    score_queue.popleft()
            image = image_new
            prev_key = key_to_press

            # diff = (np.sum(np.array(image_new) - np.array(image)) / np.sum(image)) * 100
            font = cv2.FONT_HERSHEY_SIMPLEX
            t = time.time() - t
            fps = floor(1 / t)
            cv2.putText(image_new, "FPS: "+str(fps), (10, 50), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image_new, str(floor(diff)), (100, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image_new, "Frame:"+str(curr_frame), (10, 70), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image_new, "Gen: " + str(gen), (200, 50), font, 0.5, (255, 255, 70), 1, cv2.LINE_AA)
            cv2.putText(image_new, "Profit Seq: "+str(np.sum(current_profit_sequence)), (300, 50), font, 0.5, (255, 70, 0), 1, cv2.LINE_AA)
            cv2.putText(image_new, "Data Len: " + str(data_len), (200, 70), font, 0.5, (100, 255, 200), 1, cv2.LINE_AA)

            cv2.imshow('window', np.array(image_new))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            t = time.time()

    if len(data) == 0:
        gen -= 1
        continue


    np.random.shuffle(data)

    # [print(frame[0].shape) for frame in data if frame[0].shape != (WIDTH, HEIGHT, 3)]

    batch_num = floor(len(data)/batch_size)
    for mini_batch in range(batch_num):
        data_batch = data[mini_batch*batch_size : (mini_batch+1)*batch_size]
        train = data_batch[:-int(len(data_batch) / 2)]
        test = data_batch[-int(len(data_batch) / 2):]

        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
        Y = [i[1] for i in train]

        test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
        test_Y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_X}, {'targets': test_Y}),
                snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

