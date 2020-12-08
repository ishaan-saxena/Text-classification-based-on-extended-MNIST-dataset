import emnist
from emnist import list_datasets
list_datasets()
from emnist import extract_training_samples
global images, labels
images, labels = extract_training_samples('byclass')
import matplotlib.pyplot as plt
plt.imshow(images[78])
plt.show()
import tensorflow as tf
import keras
import numpy as np
#images = np.reshape()
print(images.shape)
#plt.imshow(images[543])
#plt.show
st = 0
en = 0
bs = 697932
global model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(256 ,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(62, activation = tf.nn.softmax))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')#, matrics = ['accuracy'])
model.build()
model.summary()
def batch(beg, end):
    loc_img = images[beg:end]
    loc_lab = labels[beg:end]
    #loc_img = loc_img.reshape(-1,784)
    loc_img = tf.keras.utils.normalize(loc_img, axis = 1)
    loc_lab = tf.keras.utils.to_categorical(loc_lab,62)
    #for i in 1000:
    model.fit(loc_img,loc_lab, epochs = 1)

    
    del loc_img, loc_lab
#loc = images[0]
#plt.imshow(loc)
#plt.show()
st = 0
en = 1000
while(en<=697000):
  
    batch(st,en)
    st = st + 1000
    en = en + 1000

st = st - 1000
en = en - 1000
en = en + 932
batch(st,en)

model.save('C:\\Users\\user\\Desktop')

import cv2
#import tensorflow as tf
#import keras

vid = cv2.VideoCapture(0)

a=1
print("####################################################################")

while(True):
    a = a+1
    check, frame = vid.read()
    #print(frame)

    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('video',gr)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print("####################################################################")
print(frame)
print(a)
vid.release()
cv2.destroyAllWindows





print(frame)

frame = cv2.resize(frame, (28,28))

frame = tf.keras.utils.normalize(frame, axis = 1)


#frame.reshape((28,28))

print(frame.shape)

predic = model.predict(frame)

import numpy as np

print(np.argmax(predic))
