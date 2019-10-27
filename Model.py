from keras.models import *
from keras.layers import *



class Model:

    def __init__(self): # Notre méthode constructeur
        self.model = Sequential()
    
    def Create(self):
        self.model.add(Dense(units=64, activation='relu', input_dim=100))
        self.model.add(Dense(units=10, activation='softmax'))

    def Compile(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        
    def Train(self):
        self.model.fit(x_train, y_train, epochs=5, batch_size=32)
        #model.train_on_batch(x_batch, y_batch)

    def Predict():
        self.classes = model.predict(x_test, batch_size=128)
