from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(261,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.7, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])
    return(model)


