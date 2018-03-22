import gym
import tflearn
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from tflearn.layers.estimator import regression
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.optimizers import RMSProp


game = "BreakoutDeterministic-v4"
env = gym.make(game)
env.reset()

def get_epsilon(epsilon, gen):
    epsilon = epsilon
    epsilon -= gen * 0.009
    return epsilon

LR = 1e-3
num_games = 1000     # arbitrary number, not final
possible_actions = env.action_space.n
MODEL_NAME = 'data/Model_{}'
gamma = 0.99
epsilon = 0.7
generations = 10000	# arbitrary number, not final
height = 84
width = 84

# instead of using experience replay, i'm simply calling this function in generations to generate training data
def play4data(gen, epsilon):
    training_data = []
    for i in range(num_games):

        score = 0
        data = []
        prev_observation = []
        env.reset()
        done = False
        d = deque()

        while not done:

            # env.render()

            # if it's 0th generation, model hasn't been trained yet, so can't call predict funtion
            # or if i want to take a random action based on some fixed epsilon value
            # or if it's in later gens , but doesn't have 4 frames yet , to send to model
            if gen == 0 or len(prev_observation)==0 or np.random.rand() <= epsilon or len(d) < 4:
                theta = np.random.randn(possible_actions)
            else:
                theta = q_model.predict(np.array(d).reshape(-1, 4, height, width))[0]

            # action is a single value, namely max from an output like [0.00147357 0.00367402 0.00365852 0.00317618]
            action = np.argmax(theta)
            # action = env.action_space.sample()

            # take an action and record the results
            observation, reward, done, info = env.step(action)

            # since observation is 210 x 160 pixel image, resizing to 84 x 84
            observation = cv2.resize(observation, (height, width))

            # converting image to grayscale
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


            # this one is just total score after each game
            score += reward

	    # d is a queue of 4 frames that i pass as an input to the model
            d.append(prev_observation)
            if len(d) > 4:
                d.popleft()
            
            if len(prev_observation) > 0 and len(d) == 4 :
                training_data.append([d, action, reward, done])

            prev_observation = observation

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
                         learning_rate=lr, batch_size = 32, name='targets')

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

    targets = np.zeros((len(training_data), possible_actions))
    state = np.array([i[0] for i in training_data]).reshape(-1, 4, height, width)
    action = [i[1] for i in training_data]
    reward = np.sign([i[2] for i in training_data])
    done = [i[3] for i in training_data]

    for i in range(len(training_data)):
        with graph2.as_default():
            targets[i] = q_model.predict(state[i-1 if i > 0 else 0].reshape(-1, 4, width, height))

        with graph1.as_default():
            Q_sa = target_model.predict(state[i].reshape(-1, 4, width, height))

        if done:
            targets[i, action[i]] = reward[i]
        else:
            targets[i, action[i]] = reward[i] + gamma * np.max(Q_sa)

#         print('targets {0} Q_sa {1}'.format(targets[i], Q_sa))

	# X is the queue of 4 frames
    with graph2.as_default():
        q_model.fit({'input': state}, {'targets': targets}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')

# repeating the whole process in terms of generations
# training again and again after playing for set number of games
for gen in range(generations):
    
    epsilon = get_epsilon(epsilon, gen)
    
    training_data =  play4data(gen, epsilon)
    train2play(training_data)

    with graph2.as_default():
        q_model.save(MODEL_NAME.format(game))

    if gen % 10 == 0:
        with graph1.as_default():
            target_model.load(MODEL_NAME.format(game))
