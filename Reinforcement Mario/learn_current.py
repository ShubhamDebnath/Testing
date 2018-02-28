import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import deque
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, F1, A, W, S, D, T
# from minibatch_generator import *
# from conv_filter import inception_v3 as googlenet
from conv_filter import simple_model
import time
import cv2
from math import floor

num_games = 20
num_gens = 100
# idle_sequence = 80
# min_score_sum = 10
gamma = 0.9
epsilon = 0.7
possible_keys = [W, A, S, D]
WIDTH = 182
HEIGHT = 182
LR = 1e-3
batch_size = 50
EPOCHS = 1
prev_frames = 4
data = []
data_len = 0
filename = "data/data_{}.npy"
MODEL_NAME = "Play_Mario.model"
# model = googlenet(WIDTH, HEIGHT, 3, LR, 4, model_name=MODEL_NAME)
model = simple_model(WIDTH, HEIGHT, prev_frames, LR, 4, model_name=MODEL_NAME)

def right_movement(image_new, image):
    diff = (np.sum(image_new - image) / np.sum(image)) * 100
    if diff > 5:
        return diff
    else:
        return 0


def button_mash(key, prev_key):
    if key == W:
        ReleaseKey(prev_key)
        timeout = time.time() + 0.08
        PressKey(key)
        while time.time() < timeout:{}
    else:
        ReleaseKey(prev_key)
        PressKey(key)

print("Starting")
for i in range(4)[::-1]:
    print(i)
    time.sleep(1)

# tf.reset_default_graph()

# let's try with 100 generations
for gen in range(num_gens):

    if gen>0:
        model.load(MODEL_NAME)

    
    # 20 games in one generation
    for games in range(num_games):
        print('currently playing game number {0} in generation {1}'.format(games, gen))

        PressKey(T)
        print('pressed T')
        time.sleep(0.0001)
        ReleaseKey(T)
        time.sleep(0.3)
        PressKey(F1)
        print('pressed F1')
        time.sleep(0.0001)
        ReleaseKey(F1)
        # curr_frame = 0
        image = grab_screen(region=(0, 100, WIDTH-1, HEIGHT+99))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image_queue = deque()
        # array_sum = 0
        score = 0
        t = time.time()
        prev_key = D
        # total_score = 0

        # start loop for 400 in-game seconds
        while (True):
            # this group belongs to taking a screen shot of current frame and changing color
            # curr_frame += 1

            # this group is for sending the image as input and pressing the key
            if gen == 0 or np.random.rand() <= epsilon or len(image_queue) < prev_frames:
                theta = np.random.randn(4)
            else:
                image_queue_array = np.array(image_queue).reshape(WIDTH, HEIGHT, prev_frames)
                theta = model.predict([image_queue_array])[0]


            key = np.argmax(theta)
            key_to_press = possible_keys[key]
            button_mash(key_to_press, prev_key)
            image_new = grab_screen(region=(0, 100, WIDTH - 1, HEIGHT + 99))
            image_new = cv2.cvtColor(image_new, cv2.COLOR_BGRA2GRAY)
            score = right_movement(image_new, image)
            if np.average(np.array(image_new)) > 5:
            	image_queue.append(image_new)


            if len(image_queue) > prev_frames:
                image_queue.popleft()


            if gen == 0 or len(image_queue) < prev_frames:
                Q_sa = np.zeros(4)
            else:
                image_queue_array = np.array(image_queue).reshape(WIDTH, HEIGHT, prev_frames)
                Q_sa = model.predict([image_queue_array])[0]


            result = (theta == np.amax(theta)) * 1 * score + (gamma * np.amax(Q_sa))


            if len(image_queue) == prev_frames and np.sum(result) > 0:
                image_queue_array = np.array(image_queue).reshape(-1, WIDTH, HEIGHT, prev_frames)
                data.append([image_queue_array, result])
                data_len += 1


            if data_len % batch_size == 0 and data_len != 0:
                np.save(filename.format(int(data_len/batch_size)), data)
                data = []
            # print('pressed ' + str(result))

            # creating a list of all possible the in-game frames which is input for CNN
            # if curr_frame not in img_index:
            #     img_index.append(curr_frame)
            # NOTE: list of same images created again and again for each game
            # current_profit_sequence = score + current_profit_sequence*0.5
            # if current_profit_sequence < 0:
            #     current_profit_sequence = 0
            # if current_profit_sequence > 5:
            #     data.append([image_new, result])
            #     data_len +=1

            # if curr_frame >= 200:
            #     score_queue.append(score)
            #     if len(score_queue) == idle_sequence:
            #         temp_array = np.array(score_queue)
            #         array_sum = np.sum(temp_array)
            #         if array_sum < min_score_sum:
            #             break
            #         score_queue.popleft()

            if np.average(np.array(image_new)) < 5 and data_len>0:
                print('data len : ' + str(data_len))
                break


            image = image_new
            prev_key = key_to_press

            # diff = (np.sum(np.array(image_new) - np.array(image)) / np.sum(image)) * 100
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # t = time.time() - t
            # fps = floor(1 / t)
            # cv2.putText(image_new, "FPS: "+str(fps), (10, 50), font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image_new, str(floor(diff)), (100, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image_new, "Frame:"+str(curr_frame), (10, 70), font, 0.2, (255, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(image_new, "Gen: " + str(gen), (100, 50), font, 0.2, (255, 255, 70), 0.5, cv2.LINE_AA)
            # cv2.putText(image_new, "Result: "+str(result), (100, 50), font, 0.2, (255, 70, 0), 1, cv2.LINE_AA)
            # cv2.putText(image_new, "Data Len: " + str(data_len), (200, 70), font, 0.2, (255, 255, 200), 1, cv2.LINE_AA)
            #
            # cv2.imshow('window', np.array(image_new))
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
            #
            # t = time.time()

    # if len(data) == 0:
    #     gen -= 1
    #     continue

    data = []
    for _ in range(EPOCHS):
        for batch_num in range(200):
            try:
                data_batch = np.load('data/data_{}.npy'.format(batch_num))

                np.random.shuffle(data_batch)

    # [print(frame[0].shape) for frame in data if frame[0].shape != (WIDTH, HEIGHT, 3)]

    # batch_num = floor(len(data)/batch_size)
    # for mini_batch in range(batch_num):
                train = data_batch[:-25]
                test = data_batch[-25:]

                X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, prev_frames)
                Y = [i[1] for i in train]

                test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, prev_frames)
                test_Y = [i[1] for i in test]

                model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_X}, {'targets': test_Y}),
                snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)

            except:
                pass

            finally:
                model.save(MODEL_NAME)

