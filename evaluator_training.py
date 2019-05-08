# CAUTION! TERRIBLE RAM MANAGEMENT IN PLACE.
# TODO: Implement tensorflow's dataset pipeline
# so the entire file isn't read into memory.

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import layers
import argparser
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import xgboost as xgb

piece_id = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

scoredGames = np.array(pd.read_csv('./training_data/fics_0_20000.csv', header=None))

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
        outMatrix.append(zslice.reshape((8,8)).astype(int))
    outMatrix.append(np.full((8,8), whiteTurn))
    
    return(np.array(outMatrix))



x = np.array([stateVec2Mat(i) for i in scoredGames])

def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(2048, activation='relu', input_shape=(13, 8, 8)))
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


checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch.
    period=10)




EPOCHS = 70

history = model.fit(
  x, y,
  epochs=EPOCHS, 
  validation_split = 0.2, 
  verbose=1,
  callbacks = [cp_callback])


model.load_weights('./training/cp-0008.ckpt')






dataDmatrix = xgb.DMatrix(data=x,label=y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1.0, learning_rate = 0.1,
                max_depth = 20, alpha = 10, n_estimators = 1000, nthread=8)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


