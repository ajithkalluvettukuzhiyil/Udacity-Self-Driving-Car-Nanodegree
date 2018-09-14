###
In this quiz you will build a multi-layer feedforward neural network to classify traffic sign images using Keras.

Set the first layer to a Flatten() layer with the input_shape set to (32, 32, 3).
Set the second layer to a Dense() layer with an output width of 128.
Use a ReLU activation function after the second layer.
Set the output layer width to 5, because for this data set there are only 5 classes.
Use a softmax activation function after the output layer.
Train the model for 3 epochs. You should be able to get over 50% training accuracy.
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

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#in[4]:

# TODO: Build the Fully Connected Neural Network in Keras Here
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))


#in[5]:

# preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
# TODO: change the number of training epochs to 3
history = model.fit(X_normalized, y_one_hot, epochs=3, validation_split=0.2)

#in[6]:

### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(model, history)
except Exception as err:
    print(str(err))


