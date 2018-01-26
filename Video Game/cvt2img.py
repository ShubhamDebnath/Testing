import numpy as np
from PIL import Image


all_data = []
for i in range(0,1):
     data = np.load('C:\\Users\\gamef\\Desktop\\training_data.npy_{}.npy'.format(i))
     for j in range(len(data)):
             all_data.append([data[j][0], data[j][1]])



for i in range(len(all_data)):
     img = Image.fromarray(all_data[i][0], mode = 'L')
     img.save('C:\\Users\\gamef\\Desktop\\self_drive\\test2\\{}.png'.format(i))