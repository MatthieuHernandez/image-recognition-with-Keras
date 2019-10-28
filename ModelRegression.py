from keras.models import *
from keras.layers import *
import keras

class ModelRegression:

    def __init__(self):
        self.model = Sequential()


    def Create(self):
        #Model
        self.model.add(Conv2D(100, kernel_size=4, activation='relu', input_shape=(20, 20, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='tanh'))
        self.model.add(Dense(1, activation='sigmoid'))

        #Compile
        self.model.compile(loss='mean_squared_error',
              optimizer='sgd',#adam
              metrics=['accuracy', 'mae']) #metrics=['mse','mae']
        
    def Train(self, set, epochs):
        #Fit
        data = set[0]
        #print(data.shape)
        labels = set[1]
        #print(labels.shape)
        history = self.model.fit(data, labels, epochs=epochs, verbose = 0)#10 verbose = 2
        return history
        
        #Train
        #self.model.train_on_batch(labels, datalabels, batch_size=1)


    def Evaluate(self, set) :
        #Evaluate
        data = set[0]
        labels = set[1]
        score = self.model.evaluate(data, labels, verbose=0, use_multiprocessing=False)
        return score


    def Predict(self, x):
        return self.model.predict(x)
