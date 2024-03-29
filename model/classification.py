import keras
from keras.layers import *
from keras.models import *


class CustomModel:

    def __init__(self):
        self.model = Sequential()

    def create(self):
        # Model
        self.model.add(Dense(150, activation='tanh'))
        self.model.add(Dense(80, activation='tanh'))
        self.model.add(Dense(10, activation='sigmoid'))

    def train(self, dataset, optimizer, epochs, verbose=0):
        # Compile
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        data = dataset[0]
        # print(data.shape)
        labels = keras.utils.to_categorical(dataset[1], 10)
        # print(labels.shape)
        history = self.model.fit(data, labels, epochs=epochs, verbose=verbose)
        return history

        # Train
        #self.model.train_on_batch(labels, datalabels, batch_size=1)

    def evaluate(self, dataset):
        # Evaluate
        data = dataset[0]
        labels = keras.utils.to_categorical(dataset[1], 10)
        score = self.model.evaluate(
            data, labels, verbose=0, use_multiprocessing=False)
        return score

    def predict(self, x):
        return self.model.predict(x)
