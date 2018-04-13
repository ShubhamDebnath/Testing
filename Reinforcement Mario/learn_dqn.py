import gym
import tflearn
import numpy as np
import cv2
import random
import math
import tensorflow as tf
from collections import deque
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.optimizers import RMSProp


game = "BreakoutDeterministic-v4"   # this one skips 4 frames every time
# game = 'Breakout-v0'              # this one skips between 2 and 5 frames every time
env = gym.make(game)
env.reset()


LR = 0.00025
num_games = 1000     # arbitrary number, not final
possible_actions = env.action_space.n
MODEL_NAME = 'data/Model_{}'
gamma = 0.99
epsilon = 1
generations = 10000	# arbitrary number, not final
height = 84
width = 84
batch_size = 32

def get_epsilon(epsilon):
    if epsilon == 0.1:
        return epsilon
    epsilon -= epsilon * 0.009
    return epsilon

def pre_process_image(observation):
    # since observation is 210 x 160 pixel image, resizing to 84 x 84
    observation = cv2.resize(observation, (height, width))

    # converting image to grayscale
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return observation

def getRandomMiniBatches(X, minibatch_size ):
    # total training examples
    m = len(X)
    # we will store the mini batches as tuple pair in a list
    mini_batches = []
    
    ## First we need to shuffle the training examples
    np.random.shuffle(X)
    
    # find the total no. of mini batches that can be formed from the training examples
    num_minibatches = math.floor(m/minibatch_size) 
    
    # now partition the original training set into mini batches
    # we make mini batches till the time complete mini batches can be made
    for i in range(num_minibatches):
        minibatch_X = X[i*minibatch_size: (i+1)*minibatch_size]
        
        minibatch = (minibatch_X)
        mini_batches.append(minibatch)
        
    # if number of minibatches that can be made is not a multiple of 'm'
    if m % minibatch_size != 0:
        minibatch_X = X[num_minibatches*minibatch_size: m-1]
        
        minibatch = (minibatch_X)
        mini_batches.append(minibatch)
        
    return mini_batches


# instead of using experience replay, i'm simply calling this function in generations to generate training data
def play4data(gen, epsilon, num_games):
    training_data = []

    for i in range(num_games):

        score = 0
        data = []
        prev_observation = []
        env.reset()
        done = False
        d = []
        d_next = []
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # since observation is 210 x 160 pixel image, resizing to 84 x 84
        observation = cv2.resize(observation, (height, width))

        # converting image to grayscale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        for _ in range(4):
            # observation = pre_process_image(observation)
            d.append(observation)
        d_next = d

        while not done:

            # env.render()

            # if i want to take a random action based on some fixed epsilon value
            # or if it's in gen 0, because paper said to run random 50k steps at the start
            if gen == 0 or np.random.rand() <= epsilon:
                theta = np.random.randn(possible_actions)
            else:
                theta = q_model.predict(np.array(d).reshape(-1, 4, height, width))[0]


            # action is a single value, namely max from an output like [0.00147357 0.00367402 0.00365852 0.00317618]
            action = np.argmax(theta)
            # action = env.action_space.sample()

            # take an action and record the results
            observation, reward, done, info = env.step(action)

            # observation = pre_process_image(observation)

            # since observation is 210 x 160 pixel image, resizing to 84 x 84
            observation = cv2.resize(observation, (height, width))

            # converting image to grayscale
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


            d_next[:3] = d[1:]
            d_next[3] = observation

            # this one is just total score after each game
            score += reward

            training_data.append([d, action, reward, done, d_next])

            d = d_next


            if done:
                break

        print('gen {1} game {0}: '.format(i, gen) + str(score))

        
    env.reset()
    return training_data


# exact model described in DeepMind paper, just added a layer to end for 18 to 4
def simple_model(num_frames, width, height, lr, output=9, model_name='intelAI.model'):
    network = input_data(shape=[None, num_frames, width, height], name='input')
    conv1 = conv_2d(network, 8, 32,strides=4, activation='relu', name='conv1')
    conv2 = conv_2d(conv1, 4, 64, strides=2, activation='relu', name='conv2')
    conv3 = conv_2d(conv2, 3, 64, strides=1, activation='relu', name='conv3')
    fc4 = fully_connected(conv3, 512, activation='relu')
    # fc5 = fully_connected(fc4, 18, activation='relu')
    fc6 = fully_connected(fc4, output, activation='relu')
    
    # rmsprop = RMSProp(learning_rate=0.1, decay=0.99, momentum=0.0, epsilon=1e-6)
    network = regression(fc6, optimizer='adam',
                         loss='mean_square',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network,
                        max_checkpoints=0, tensorboard_verbose=0, tensorboard_dir='log')
    return model


# defining/ declaring the model
graph1 = tf.Graph()
with graph1.as_default():
    target_model = simple_model(4, width, height, LR, possible_actions)
graph2 = tf.Graph()
with graph2.as_default():
    q_model = simple_model(4, width, height, LR, possible_actions)

# this function is responsible for training the model
def train2play(training_data):

    mini_batches = getRandomMiniBatches(training_data, minibatch_size = batch_size)

    for mini_batch in mini_batches:
        targets = np.zeros((batch_size, possible_actions))
        state = np.array([i[0] for i in mini_batch]).reshape(-1, 4, height, width)
        action = [i[1] for i in mini_batch]
        reward = np.sign([i[2] for i in mini_batch])
        done = [i[3] for i in mini_batch]
        state_next = np.array([i[4] for i in mini_batch]).reshape(-1, 4, height, width)

        for i in range(batch_size):
            with graph2.as_default():
                targets[i] = q_model.predict(state[i].reshape(-1, 4, width, height))

            with graph1.as_default():
                Q_sa = target_model.predict(state_next[i].reshape(-1, 4, width, height))

            if done[i]:
                targets[i, action[i]] = reward[i]
            else:
                targets[i, action[i]] = reward[i] + gamma * np.max(Q_sa)

            print('targets {0} Q_sa {1}'.format(targets[i], Q_sa))

	    # X is the list of 4 frames
        with graph2.as_default():
            q_model.fit({'input': state}, {'targets': targets}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')

# repeating the whole process in terms of generations
# training again and again after playing for set number of games
for gen in range(generations):
    
    epsilon = get_epsilon(epsilon)
    
    if gen == 0:
        training_data =  play4data(gen, epsilon, 50000)
    else:
        training_data =  play4data(gen, epsilon, num_games)

    train2play(training_data)

    # saving then copying q_model to target_model
    with graph2.as_default():
        q_model.save(MODEL_NAME.format(game))
        # print(' main model saved')


    if gen % 10 == 0:
        with graph1.as_default():
            target_model.load(MODEL_NAME.format(game))
            # print('target_model copied from main model')