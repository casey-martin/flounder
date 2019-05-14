### CAUTION: READS ENTIRE FILE INTO MEMORY.
### TODO: Read in only centipawn score and sample based off indices. 

import argparse
import csv
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, help='Input training data')
parser.add_argument('--outdir', type=str, help='Directory to which subsampled training data will be written.')
parser.add_argument('--cutoff', type=float, default=0.1, help='Percentage cutoff')
parser.add_argument('--verbose', type=bool, default=True, help='Gives updates on script progress')

args = parser.parse_args()

def main():

    if args.verbose:
        print('\nReading training data...')
    
    trainData = pd.read_csv(args.train_data, header=None) 
    #trainData = []
    #with open(args.train_data) as csvfile:
    #    readCSV = csv.reader(csvfile, delimiter=',')
    #    for row in readCSV:
    #        trainData.append(row)
    
    trainData = np.asarray(trainData)
    
    if args.verbose:
        print('\nReindexing...')
    # sort by absolute centipawn value. 
    #trainData = trainData[trainData[:,-1].argsort()][::-1]
    trainData = np.array(sorted(trainData, key=lambda row: np.abs(row[-1]), reverse=True))

    #trainData.index = range(len(trainData))
    
    thresh = round(args.cutoff * trainData.shape[0])
    
    # Save the top percent of extreme centipawn evals from the training data.
    outFrame = [trainData[:thresh]]


    # Sample the remaining data.
    #outFrame.append(trainData.iloc[thresh:].sample(n=thresh, axis=0))
    outFrame.append(trainData[np.random.choice(trainData.shape[0], thresh, replace=False), :])

    outFrame = np.concatenate(outFrame)

    myBasename = os.path.splitext(os.path.basename(args.train_data))
    myOutname = myBasename[0] + '_' +  str(args.cutoff) + myBasename[1]

    if args.verbose:
        print('\nShuffling...')
    # Shuffles the outgoing data.
    np.random.shuffle(outFrame)
    
    if args.verbose:
        print('\nSaving to file...')
    #outFrame.to_csv(os.path.join(args.outdir, myOutname), header=False, index=False)
    np.savetxt(os.path.join(args.outdir, myOutname), outFrame, delimiter=",",fmt='%i')

if __name__ == '__main__':
    main()
