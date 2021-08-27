
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization


class BrainTumorModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), padding='Same', input_shape=(128, 128, 3)))
        self.model.add(Conv2D(32, (2, 2), activation='relu', padding='Same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (2, 2), activation='relu', padding='Same'))
        self.model.add(Conv2D(64, (2, 2), activation='relu', padding='Same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2, activation='softmax'))

    def load_model(self):
        return self.model

    def summary_model(self):
        return self.model.summary()
# model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BrainTumorModel')
# model.summary()