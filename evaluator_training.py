# CAUTION! TERRIBLE RAM MANAGEMENT IN PLACE.
# TODO: Implement tensorflow's dataset pipeline
# so the entire file isn't read into memory.

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import layers
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Input training data')
parser.add_argument('--outdir', type=str, help='Directory where network weights will be saved.')
parser.add_argument('--epochs', type=int, help='Number of epochs the model will be trained for.')
parser.add_argument('--period', type=int, help='Periodicity of the checkpoint saves.')


args = parser.parse_args()

piece_id = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

scoredGames = np.array(pd.read_csv(args.train_data, header=None))

#x = np.array(scoredGames[:,:65])
y = np.array(scoredGames[:,-1])
#normalize y values
y = (y-min(y))/(max(y)-min(y)) 

def stateVec2Mat(stateVec):
    board = stateVec[:64]
    whiteTurn = stateVec[64]
    
    outMatrix = []
    for i in piece_id:
        zslice = board == i
        outMatrix.append(zslice.astype(int))
    outMatrix.append(np.array([whiteTurn]))
    
    return(np.hstack(outMatrix))



x = np.array([stateVec2Mat(i) for i in scoredGames])

def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(2048, activation='relu', input_shape=(769,)))
    model.add(layers.Dense(2048, activation=tf.nn.relu))
    model.add(layers.Dense(2048, activation=tf.nn.relu))
    model.add(tf.layers.Flatten())
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.7, decay=1e-08, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return(model)

model = build_model()


checkpoint_path = os.path.join(args.outdir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch.
    period=args.period)


history = model.fit(
  x, y,
  epochs=args.epoch, 
  validation_split = 0.2, 
  verbose=1,
  callbacks = [cp_callback])
