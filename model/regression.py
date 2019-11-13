# from tensorflow.python.keras.applications import ResNet50
# from keras import regularizers

import keras
from keras.callbacks import *
from keras.layers import *
from keras.models import *


class CustomModel:

    def __init__(self, model_type):
        self.model = Sequential()
        self.__create(model_type)

    def __create(self, model_type):

        if model_type == 'simple':
            # self.model.add(
            #    LocallyConnected2D(1, kernel_size=2, activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(500, activation='relu'))
            self.model.add(Dense(40, activation='tanh'))
            self.model.add(Dense(1, activation='sigmoid'))

        elif model_type == 'complexe':
            self.model.add(Conv2D(8, kernel_size=4, padding='same', activation='relu',
                                  bias_regularizer=keras.regularizers.l2(
                                      0.0001)
                                  ))
            self.model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
            self.model.add(Flatten())
            self.model.add(Dense(350, activation='tanh'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.20))

            self.model.add(Dense(120, activation='tanh'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.15))

            self.model.add(Dense(35, activation='sigmoid'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.10))

            self.model.add(Dense(1, activation='sigmoid'))

        elif model_type == 'pyramidnet':
            self.model.add(Conv2D(1, kernel_size=4, padding='valid', activation='relu',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=4, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=4, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=4, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=4, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=2, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Conv2D(1, kernel_size=2, padding='valid', activation='sigmoid',
                                  use_bias=True, bias_initializer='Zeros', bias_regularizer=keras.regularizers.l2(0.01)
                                  ))
            self.model.add(Flatten())
            self.model.add(Dense(8, activation='sigmoid'))
            self.model.add(Dense(1, activation='sigmoid'))

        elif model_type == 'resnet':
            for _layer in range(0, 3):
                self.model.add(Conv2D(4, kernel_size=2, padding='same', activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(
                                          0.0001)
                                      ))
            self.model.add(BatchNormalization())
            self.model.add(Flatten())
            self.model.add(Dense(120, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(90, activation='tanh'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(1, activation='sigmoid'))
        else:
            raise Exception("CustomModel must have a valid type")

    def train(self, train, test, optimizer, epochs, verbose=0):
        # Compile
        if optimizer == 'sgd':
            self.model.compile(loss='mean_squared_error',
                               optimizer=keras.optimizers.SGD(
                                   learning_rate=0.01, momentum=0.7, nesterov=False),
                               metrics=['mae'])
        if optimizer == 'adam':
            self.model.compile(loss='mean_squared_error',
                               optimizer='adam',  # adadelta
                               metrics=['mae'])

        history = self.model.fit(
            train[0], train[1],
            batch_size=16,
            callbacks=[EarlyStopping(
                monitor='val_loss', patience=8, mode='min', restore_best_weights=True)],
            epochs=epochs,
            verbose=verbose,
            validation_data=test,
            validation_freq=6,
            use_multiprocessing=True)
        return history

    def evaluate(self, dataset):
        # Evaluate
        data = dataset[0]
        labels = dataset[1]
        score = self.model.evaluate(
            data, labels, verbose=0, use_multiprocessing=False)
        return score

    def predict(self, x):
        return self.model.predict(x)
