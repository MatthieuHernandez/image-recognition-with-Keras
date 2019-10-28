from keras.models import *
from keras.layers import *
import keras

class ModelClassification:

    def __init__(self):
        self.model = Sequential()


    def Create(self):
        #Model
        self.model.add(Dense(500, activation='tanh'))
        self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dense(10, activation='sigmoid'))

        #Compile
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',#adam #sgd
              metrics=['accuracy']) #metrics=['mse','mae']
        
    def Train(self, set, epochs):
        #Fit
        data = set[0]
        #print(data.shape)
        labels = keras.utils.to_categorical(set[1], 10)
        #print(labels.shape)
        history = self.model.fit(data, labels, epochs=epochs, verbose = 0)#10 verbose = 2
        return history
        
        #Train
        #self.model.train_on_batch(labels, datalabels, batch_size=1)


    def Evaluate(self, set) :
        #Evaluate
        data = set[0]
        labels = keras.utils.to_categorical(set[1], 10)
        score = self.model.evaluate(data, labels, verbose=0, use_multiprocessing=False)
        return score


    #def Predict():
    #    self.classes = self.model.predict(x_test, batch_size=128)
