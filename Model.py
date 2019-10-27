from keras.models import *
from keras.layers import *
import keras


class Model:

    def __init__(self):
        self.model = Sequential()


    def Create(self):     
        inputs = keras.layers.Input(shape=(1200,))
        hiddenLayer1 = keras.layers.Input(shape=(32,))
        
        layer1 = keras.layers.Dense(8, activation='tanh')(inputs)       
        layer2 = keras.layers.Dense(8, activation='tanh')(hiddenLayer1)

        added = keras.layers.add([layer1, layer2])
        outputs = keras.layers.Dense(1)(added)
        model = keras.models.Model(inputs=[inputs, hiddenLayer1], outputs=outputs)


    def Compile(self):
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        
    def Train(self, set):
        data = set[0]
        #print(data.shape)
        labels = set[1]
        #print(labels.shape)
        self.model.fit(data)
        #self.model.train_on_batch(labels, datalabels)


    def Evaluate(self, set) :
        labels = set[0]
        data = set[1]
        loss_and_metrics = model.evaluate(labels, data, batch_size=128)


    #def Predict():
    #    self.classes = self.model.predict(x_test, batch_size=128)
