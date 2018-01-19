import numpy as np
from alexnet import alexnet2

HEIGHT = 225
WIDTH = 400
EPOCHS = 5
LR = 1e-3
MODEL_NAME = 'Crossout_Bot/Crossout-Bot-LR{}-EPOCHS{}.model'.format(LR, EPOCHS)

model = alexnet2(WIDTH, HEIGHT, LR, output=4)

#model.load(MODEL_NAME)

data = np.load('training_data.npy')

np.random.shuffle(data)
train = data[:-int(len(data)/2)]
test = data[-int(len(data)/2):]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch = EPOCHS,
          validation_set = ({'input': test_X}, {'targets': test_Y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/gamef/Desktop/log