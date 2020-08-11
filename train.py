import numpy as np
import numpy.random as rnd
import os
import cv2
import skimage
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Input
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers.core import Lambda
from keras import backend as K


path = './data'

def read_and_resize(img):
	img = skimage.io.imread(img)
	img = skimage.transform.resize(img,(256,256))

	return img


def get_mini_batch(batch_size=32, prob=0.5, path = path):

	persons = os.listdir(path)
	left=[]
	right=[]
	target=[]

	for r in range(batch_size):
		res = rnd.choice([0,1],p=[1-prob,prob])
		if res == 0:
			t1,t2 = tuple(rnd.choice(persons, size = 2, replace = False))
			
			t1 = os.path.join(path,t1)
			t2 = os.path.join(path,t2)
			t1,t2 = read_and_resize(t1),read_and_resize(t2)

			#t1,t2 = np.reshape(img1,(-1,256,256,3)), np.reshape(img2,(-1,256,256,3))

			left.append(t1)
			right.append(t2)
			target.append(0)
		else :
			t = rnd.choice(persons)
			#t1,t2 = tuple(rnd.choice(1,size = 2, replace = False))
			t1 = os.path.join(path,t)
			t2 = os.path.join(path,t)
			t1,t2 = read_and_resize(t1),read_and_resize(t2)

			#t1,t2 = np.reshape(img1,(-1,256,256,3)), np.reshape(img2,(-1,256,256,3))
			left.append(t1)
			right.append(t2)
			target.append(1)

	return [np.array(left),np.array(right)],np.array(target)		


			
def siamese_model(input_shape):

	left_input = Input(input_shape)
	right_input = Input(input_shape)

	model = Sequential()
	model.add(Conv2D(32, (9,9), activation = 'relu', input_shape = input_shape))
	model.add(MaxPooling2D((3,3)))

	model.add(Conv2D(64, (5,5),activation = 'relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(64, (5,5),activation = 'relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(128, (3,3),activation = 'relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Conv2D(256, (3,3),activation = 'relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Flatten())

	model.add(Dense(4096,activation='softmax'))

	encoded_l = model(left_input)
	encoded_r = model(right_input)

	L1_layer = Lambda( lambda tensors:K.abs(tensors[0]-tensors[1]))
	L1_distance = L1_layer([encoded_l,encoded_r])

	prediction = Dense(1,activation='sigmoid')(L1_distance)

	net = Model(inputs=[left_input,right_input],outputs = prediction)

	return net



model = siamese_model((256,256,3))
model.summary()

optimizer = Adam()
model.compile(loss = "binary_crossentropy", optimizer = optimizer ,metrics = ['accuracy'])

batch_size=32
loss_history = []

(images,labels)=get_mini_batch(batch_size = 64, path = path)


model.fit(images,labels,epochs = 20, batch_size=32)


model.save("one_shot_80.h5")
print("model saved successfully")


