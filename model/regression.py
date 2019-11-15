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
            self.model.add(Dense(300, activation='relu'))
            self.model.add(Dense(40, activation='tanh'))
            self.model.add(Dense(1, activation='sigmoid'))

        elif model_type == 'complexe':

            regu = keras.regularizers.l2(0.0001)

            self.model.add(
                Conv2D(12, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regu))
            self.model.add(BatchNormalization())
            self.model.add(
                Conv2D(12, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regu))
            self.model.add(BatchNormalization())
            self.model.add(AveragePooling2D(pool_size=(2, 2), padding='valid'))
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(
                Dense(256, activation='tanh', kernel_regularizer=regu))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(1, activation='sigmoid'))

        elif model_type == 'resnet':

            inputs = keras.engine.input_layer.Input()

            self.model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(
                                      0.01)
                                  ))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(
                                      0.01)
                                  ))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(
                                      0.01)
                                  ))

            self.model.add(BatchNormalization())
            self.model.add(AveragePooling2D(pool_size=(
                2, 2), padding='valid'), kernel_regularizer=keras.regularizers.l2(0.01))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'),
                           kernel_regularizer=keras.regularizers.l2(0.01))
            self.model.add(Dropout(0.4))
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

        early_stop = EarlyStopping(
            monitor='val_mae', patience=40, mode='min', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_mae', factor=0.5, patience=16, mode='min', min_delta=0.0001)

        history = self.model.fit(
            train[0], train[1],
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            epochs=epochs,
            verbose=verbose,
            validation_data=test,
            validation_freq=2,
            workers=8,
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
