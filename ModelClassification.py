from keras.models import *
from keras.layers import *
import keras

class ModelClassification:

    def __init__(self):
        self.model = Sequential()


    def Create(self):
        #Model
        self.model.add(Dense(150, activation='tanh'))
        self.model.add(Dense(80, activation='tanh'))
        self.model.add(Dense(10, activation='sigmoid'))


        
    def Train(self, set, optimizer, epochs, verbose = 0):
        #Compile
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        data = set[0]
        #print(data.shape)
        labels = keras.utils.to_categorical(set[1], 10)
        #print(labels.shape)
        history = self.model.fit(data, labels, epochs=epochs, verbose=verbose)
        return history
        
        #Train
        #self.model.train_on_batch(labels, datalabels, batch_size=1)


    def Evaluate(self, set) :
        #Evaluate
        data = set[0]
        labels = keras.utils.to_categorical(set[1], 10)
        score = self.model.evaluate(data, labels, verbose=0, use_multiprocessing=False)
        return score


    def Predict(self, x):
        return self.model.predict(x)
