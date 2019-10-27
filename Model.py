from keras.models import *
from keras.layers import *

model = Sequential()

def CreateModel():
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

