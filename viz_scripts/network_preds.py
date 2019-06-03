from scipy import stats
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.python.keras import layers
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--state_vec', type=str, help='Path to test data. Should be in the same format as training data.')
parser.add_argument('--model', type=str, help='Path to trained model')
parser.add_argument('--outdir', type=str, help='Directory to which plot will be saved.')
args = parser.parse_args()


piece_id = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]

def stateVec2Mat(stateVec):
    board = stateVec[:64]
    whiteTurn = stateVec[64]
    
    outMatrix = []
    for i in piece_id:
        zslice = board == i
        outMatrix.append(zslice.astype(int))
    outMatrix.append(np.array([whiteTurn]))
    
    return(np.hstack(outMatrix))

def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(2048, activation='relu', input_shape=(769,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2048, activation=tf.nn.relu, input_shape=(769,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1050, activation=tf.nn.relu, input_shape=(769,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5))
    model.add(layers.Dense(1))
    #model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.7, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return(model)


def main():
    boardStates = np.asarray(pd.read_csv(args.state_vec))
    x = np.array([stateVec2Mat(i) for i in boardStates])
    
    y = boardStates[:,-1]
    y[y < -10000] = -10000
    y[y > 10000] = 10000
    y = (y-np.min(y))/(np.max(y)-np.min(y))
    y = y.reshape((y.shape[0],-1))
    model = build_model()
    model.load_weights(args.model)
    
    ypred = model.predict(x)
    
    #fit = np.polyfit(y[:,0],ypred[:,0], 1)
    #fit_fn = np.poly1d(fit)
    mse = mean_squared_error(y, ypred)   

 
    fig, ax = plt.subplots(nrows=1, ncols=1)
    hb = ax.hexbin(np.array(y), np.array(ypred), gridsize=50, bins='log',cmap=plt.cm.bone)
    cb = fig.colorbar(hb, ax = ax)
    cb = cb.set_label('log10(N)')
    #ax.plot(y, fit_fn(y), '--k')    

    fig.suptitle(args.model,size=20)
    ax.set_xlabel('Stockfish Evaluation',size=16)
    ax.set_ylabel('ANN Evaluation',size=16)
    ax.text(0.7, 0.1, 'MSE:' + str(round(mse, 5)),bbox=dict(facecolor='white', edgecolor='black', pad=2))

    outName = os.path.basename(args.model) + '.png'

    fig.savefig(os.path.join(args.outdir, outName))
    plt.close()

if __name__ == '__main__':
    main()
