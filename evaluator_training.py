# Bootleg tf dataset pipeline in place. 
# for tf > 1.9, keras models can directly accept tf datasets.
# my cuda/cudnn drivers and utilities are incompatible 
# with current tf versions. Will update this script once I 
# can find the courage to try and reinstall nvidia software.
from scipy.stats import rankdata
from subprocess import call
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import argparse
import h5py
import gc
import numpy as np
import os
import tensorflow as tf

def mapping_to_target_range( x, target_min=-150, target_max=150 ) :
    x02 = K.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min


def rescale(cp, maxCp = 150, minCp = -150):
    centipawn_norm = np.clip(cp, minCp, maxCp) 
    centipawn_norm = (centipawn_norm - minCp)/(maxCp - minCp)
    return(centipawn_norm)


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


class Trainer:
    def __init__(self, model, train_folder, test_data, batch_size):
        self.model = model
        self.train_folder = train_folder
        self.test_data = test_data
        self.batch_size = batch_size
        self.train_list = self.count_training()
        self.testX, self.testY = self.get_validation()

    def count_training(self):
        myFiles = os.listdir(self.train_folder)
        obsCount = 0
        train_list = []

        for hfName in myFiles:
            hfFullPath = os.path.join(self.train_folder, hfName)
            with h5py.File(hfFullPath) as hf:

               for i in hf['labels'].keys():
                    obsCount += hf['labels'][i][()].shape[0]
                    train_list.append((hfFullPath, i))
        self.obsCount = obsCount                    
        return(train_list)

    def get_validation(self):
        testX = []
        testY = []
        with h5py.File(args.test_data) as hf:
            for i in hf['labels'].keys():
                tmpX = np.array(hf['boardStates'][i][()])
                tmpY = np.array(hf['labels'][i][()])
                
                testX.append(tmpX)
                testY.append(tmpY)

        testX = np.concatenate(testX).astype('bool')
        testY = rescale(np.concatenate(testY).astype(np.float32))

        return(testX, testY)

    def shuffle_order(self):
        np.random.shuffle(self.train_list)
       
    def fit_to_training(self):
        for myhf, mykey in self.train_list:
            with h5py.File(myhf) as hf:
                print(myhf, mykey)
                tmpBoardStates = hf['boardStates'][mykey][()]
                tmpLabels = hf['labels'][mykey][()]
                # give more importance to extreme positions because they are more rare.
                # weights might need to be integer values. When passing floats, got nans during training.
                tmpSampleWeights = rankdata(np.abs(tmpLabels))/len(tmpLabels)
                tmpSampleWeights = np.round(np.log(tmpSampleWeights/np.min(tmpSampleWeights))) + 1
                tmpLabels = rescale(tmpLabels)
                with open(os.path.join(args.outdir, 'batch_length.txt'), 'a+') as f:
                    f.write(str( len(tmpLabels)/float(self.obsCount) ) + '\n')
            
                self.model.fit(
                  x = tmpBoardStates, y = tmpLabels,
                  validation_data = (self.testX, self.testY),
                  sample_weight = tmpSampleWeights,
                  batch_size = self.batch_size,
                  epochs=1, 
                  verbose=args.verbose,
                  callbacks = [csv_logger],
                  shuffle = True)



parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', type=str, help='Input training data')
parser.add_argument('--test_data', type=str, required=True, help='Input validation data')
parser.add_argument('--outdir', type=str, required=True,  help='Directory where network weights will be saved.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs the model will be trained for.')
parser.add_argument('--period', type=int, help='Periodicity of the checkpoint saves.')
parser.add_argument('--weights', type=str, default = None,
                       help='Path to pretrained weights. Omit flag if making a new model. Default=None')
parser.add_argument('--batch_size', type=int, default=256, help='Training batch size. Default=256')
parser.add_argument('--buffer_size', type=int, default=5620, help='Prefetch buffer size. Default=2560')
parser.add_argument('--verbose', type=int, default=1, help='0: silent; 1: verbose output; 2: Goldilocks. Default=2')

args = parser.parse_args()

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

csv_logger = tf.keras.callbacks.CSVLogger(args.outdir + "model_history_log.csv", append=True)


print('Reading training data...')
myTrainer = Trainer(model, args.train_folder, args.test_data, args.batch_size)

print('Found', myTrainer.obsCount, 'training examples.\n')

print('Found', myTrainer.testY.shape[0], 'test examples.\n\n\n')


for i in range(args.epochs):

    myTrainer.shuffle_order()
    myTrainer.fit_to_training()
    print('Epoch', i+1, 'of', args.epochs, 'complete. Saving model.')
    myTrainer.model.save_weights(os.path.join(args.outdir, "model.h5"))


