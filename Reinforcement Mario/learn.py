import numpy as np
import tensorflow as tf
from collections import deque
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, F1, A, W, S, D
from minibatch_generator import *
from conv_filter import *
import time
import cv2

learning_rate = 1e-8
num_epochs = 100
minibatch_size = 12
img_index = []
img_list = []
total_frames = 4100
num_games = 50
num_gens = 100
idle_sequence = 60
min_score_sum = 10
key_list = [A, D, W, S]


def right_movement(image_new, image):
    diff = (np.sum(image_new - image) / np.sum(image)) * 100
    if diff > 5:
        return diff
    else:
        return 0


def button_mash(key):
    PressKey(key)
    timeout = time.time() + 0.16
    while (time.time() < timeout): {}
    ReleaseKey(key)

# let's try with 100 generations
for gen in range(num_gens):

    score_matrix = np.zeros((4, total_frames))

    # 50 games in one generation
    for games in range(num_games):
        print('currently playing game number {0} in generation {1}'.format(games, gen))

        if gen == 0:
            theta = np.random.randn(4, total_frames)

        PressKey(F1)
        time.sleep(0.01)
        ReleaseKey(F1)
        image = np.array(grab_screen(region=(0, 60, 512, 444)))
        score_queue = deque([])

        # start loop for 400 in-game seconds
        for curr_frame in range(total_frames):
            image_new = np.array(grab_screen(region = (0, 60, 512, 444)))
            # canny_image = cv2.Canny(image_new, threshold1=200, threshold2=300)
            # tf_tensor = tf.convert_to_tensor(np.array([1, image_new]), dtype = tf.float64)
            # csonv_image = conv_model(tf_tensor)
            key = np.argmax(theta[:, curr_frame])
            key_to_press = key_list[key]
            button_mash(key_to_press)
            score = right_movement(image_new, image)
            if curr_frame >= 5:
                score_queue.append(score)
                if len(score_queue) == idle_sequence:
                    temp_array = np.array(score_queue)
                    array_sum = np.sum(temp_array)
                    if array_sum < min_score_sum:
                        continue
                    score_queue.popleft()
            if score_matrix[key, curr_frame] < score:
                score_matrix[key, curr_frame] = score * theta[key, curr_frame]
            if curr_frame not in img_index:
                img_index.append(curr_frame)
                img_list.append(image_new)
            image = image_new

    profit = sum(score_matrix)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).maximize(profit)
    init = tf.global_variables_initializer()
    seed = 3

    X = tf.placeholder(tf.float64, img_list.shape, name = 'X')   # shape to be decided yet
    Y = tf.placeholder(tf.float64, theta.shape, name = 'Y')
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_profit = 0.
            num_minibatches = int(total_frames / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(img_list, theta, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, temp_profit = sess.run([optimizer, profit], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_profit += temp_profit / num_minibatches

            # Print the cost every epoch
            if epoch % 5 == 0:
                print ("Profit after epoch %i: %f" % (epoch, minibatch_profit))
            if epoch % 1 == 0:
                profits.append(minibatch_profit)