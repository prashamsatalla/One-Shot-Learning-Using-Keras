import numpy as np
import os
import skimage 
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy


model = load_model("one_shot_80.h5")

model.compile(loss = "binary_crossentropy", optimizer = Adam() ,metrics = ['accuracy'])

def read_and_resize(img):
	img = skimage.io.imread(img)
	img = skimage.transform.resize(img,(256,256))
	img = np.reshape(img,[-1,256,256,3])

	return img
img1 = read_and_resize("./data/Obama.jpg")
img2 = read_and_resize("./test/Prash.jpg")

classes = model.predict([img1,img2])

print(classes)


