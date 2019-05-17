# Bootleg tf dataset pipeline in place. 
# for tf > 1.9, keras models can directly accept tf datasets.
# my cuda/cudnn drivers and utilities are incompatible 
# with current tf versions. Will update this script once I 
# can find the courage to try and reinstall nvidia software.

from pgn_splitter import numLines
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf

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

       


def stateVec2Mat(stateVec):

    piece_id = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    minCp = -10000
    maxCp = 10000
    
    board = stateVec[:64]
    whiteTurn = tf.reshape(tf.to_int32(stateVec[64]), [-1])
    outMatrix = tf.concat([tf.to_int32(tf.equal(board,i)) for i in piece_id],axis=0)
    outMatrix = tf.concat([outMatrix, whiteTurn], axis=0)
    #univariate centipawn score.
    
    centipawn_norm = tf.reshape(tf.clip_by_value(stateVec[-1], minCp, maxCp), [-1])
    centipawn_norm = (centipawn_norm - minCp)/(maxCp - minCp)
     
    return(outMatrix, centipawn_norm)



def make_tld(csv_filename, header_lines, delim, batch_size):
    dataset = tf.data.TextLineDataset(filenames=csv_filename).skip(header_lines)

    def parse_csv(line):
        cols_types = [[]] * 66
        columns = tf.decode_csv(line, record_defaults=cols_types, field_delim=delim)
        return(stateVec2Mat(tf.stack(columns)))

    dataset = dataset.map(parse_csv).batch(batch_size)

    return(dataset)
 

def tfdata_generator(csv_filename, header_lines, delim, batch_size):
    dataset = tf.data.TextLineDataset(filenames=csv_filename).skip(header_lines)

    def parse_csv(line):
        cols_types = [[]] * 66
        columns = tf.decode_csv(line, record_defaults=cols_types, field_delim=delim)
        return(stateVec2Mat(tf.stack(columns)))

    dataset = dataset.map(parse_csv)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=batch_size)
    iterator = dataset.make_one_shot_iterator()

    next_batch = iterator.get_next()
    # https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras
    while True:
        yield K.get_session().run(next_batch)
    
    #return(dataset)
     
parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Input training data')
parser.add_argument('--test_data', type=str, help='Input validation data')
parser.add_argument('--outdir', type=str, help='Directory where network weights will be saved.')
parser.add_argument('--epochs', type=int, help='Number of epochs the model will be trained for.')
parser.add_argument('--period', type=int, help='Periodicity of the checkpoint saves.')
parser.add_argument('--weights', type=str, default = None,
                       help='Path to pretrained weights. Omit flag if making a new model. Default=None')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size. Default=128')
parser.add_argument('--verbose', type=int, default=2, help='0: silent; 1: verbose output; 2: Goldilocks. Default=2')


args = parser.parse_args()


dataset = tfdata_generator(csv_filename=args.train_data, header_lines=0, delim=',', batch_size=args.batch_size)
testset = tfdata_generator(csv_filename=args.test_data, header_lines=0, delim=',', batch_size=args.batch_size)
trainingSize = numLines(args.train_data)
testSize = numLines(args.test_data)

if args.weights is None:
    model = build_model()
else:
    model = build_model()
    model.load_weights(args.weights)


checkpoint_path = os.path.join(args.outdir, "cp-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every epoch.
    period=args.period)


model.fit_generator(
  dataset,
  validation_data=testset,
  workers=0,
  steps_per_epoch=trainingSize // args.batch_size,
  validation_steps=testSize // args.batch_size,
  epochs=args.epochs, 
  verbose=args.verbose,
  callbacks = [cp_callback])
