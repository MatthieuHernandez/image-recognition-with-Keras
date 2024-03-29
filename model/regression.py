# from tensorflow.python.keras.applications import ResNet50
# from keras import regularizers

import keras
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.regularizers import *


class CustomModel:

    def __init__(self, model_type):
        self.__create(model_type)

    def __create(self, model_type):
        inputs = keras.engine.input_layer.Input(shape=(20, 20, 1))

        if model_type == 'simple':
            model = LocallyConnected2D(1, kernel_size=4, activation='relu')(inputs)
            model = Flatten()(model)
            model = Dense(200, activation='sigmoid')(model)
            model = Dense(1, activation='sigmoid')(model)

        elif model_type == 'complexe':
            model = Conv2D(12, kernel_size=4, padding='same', activation='relu', bias_regularizer=l2(0.0001))(inputs)
            model = MaxPooling2D(pool_size=(2, 2), padding='valid')(model)
            model = Flatten()(model)
            model = Dense(450, activation='tanh', bias_regularizer=l2(0.0001))(model)
            model = BatchNormalization()(model)
            model = Dropout(0.2)(model)
            model = Dense(100, activation='tanh', bias_regularizer=l2(0.0001))(model)
            model = BatchNormalization()(model)
            model = Dropout(0.15)(model)
            model = Dense(35, activation='tanh', bias_regularizer=l2(0.0001))(model)
            model = BatchNormalization()(model)
            model = Dropout(0.1)(model)
            model = Dense(1, activation='sigmoid')(model)

        elif model_type == 'resnet':

            first = self.__resnet_layer(inputs, 16)
            model = self.__resnet_layer(first, 16)
            model = self.__resnet_layer(model, 32)
            model = self.__resnet_layer(model, 32, first)
            model = self.__resnet_layer(model, 32)
            model = self.__resnet_layer(model, 64)
            model = self.__resnet_layer(model, 64, first)
            model = self.__resnet_layer(model, 64)
            model = self.__resnet_layer(model, 128)
            model = self.__resnet_layer(model, 128, first)
            model = self.__resnet_layer(model, 128)

            model = AveragePooling2D(pool_size=(4, 4), padding='valid')(model)
            model = Flatten()(model)
            model = Dense(256, activation='tanh', kernel_regularizer=l2(0.0001))(model)
            model = Dropout(0.4)(model)
            model = Dense(1, activation='sigmoid')(model)
        else:
            raise Exception("CustomModel must have a valid type")
        self.model = Model(inputs=inputs, outputs=model)

    def __resnet_layer(self, model, size, inputs=None):
        model = Conv2D(size, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.0001))(model)
        if inputs is not None:
            model = concatenate([model, inputs])
        return model

    def train(self, train, test, optimizer, epochs, verbose=0):
        # Compile
        if optimizer == 'sgd':
            self.model.compile(loss='mean_squared_error',
                               optimizer=keras.optimizers.SGD(
                                   learning_rate=0.01, momentum=0.4, nesterov=False),
                               metrics=['mae'])

            early_stop = EarlyStopping(
                monitor='val_mae', patience=30, mode='min', restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(
                monitor='mae', factor=0.8, patience=5, mode='min')
            history = self.model.fit(
                train[0], train[1],
                batch_size=2,
                callbacks=[early_stop, reduce_lr],
                epochs=epochs,
                verbose=verbose,
                validation_data=test,
                validation_freq=1,
                workers=8,
                use_multiprocessing=True)

        if optimizer == 'adam':
            self.model.compile(loss='mean_squared_error',
                               optimizer='adam',  # adadelta
                               metrics=['mae'])

            early_stop = EarlyStopping(
                monitor='val_mae', patience=125, mode='min', restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(
                monitor='mae', factor=0.8, patience=20, mode='min')

            history = self.model.fit(
                train[0], train[1],
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                epochs=epochs,
                verbose=verbose,
                validation_data=test,
                validation_freq=1,
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
