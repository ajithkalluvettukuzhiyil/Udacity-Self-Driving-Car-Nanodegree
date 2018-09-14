###
Build from the previous network ( Keras_nn.py).
Add a convolutional layer with 32 filters, a 3x3 kernel, and valid padding before the flatten layer.
Add a ReLU activation after the convolutional layer.
Train for 3 epochs again, should be able to get over 50% accuracy.

###

#in[1]:

import pickle
import numpy as np
import tensorflow as tf

# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)
    
    
#in[2]:

# split data
X_train, y_train = data['features'], data['labels']

#in[3]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D

#in[4]:

# TODO: Build Convolutional Neural Network in Keras Here

model = Sequential()


model.add(Conv2D(32,(3,3),
                 padding='valid',
                input_shape=(32,32,3)))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dense(5))

model.add(Activation('softmax'))


#in[5]:

# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

# compile and train model
# Training for 3 epochs should result in > 50% accuracy
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)

#in[6]:

### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(model, history)
except Exception as err:
    print(str(err))
