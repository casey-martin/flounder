import argparse
import h5py
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Input training data')
parser.add_argument('--outdir', type=str, required=True,  help='Directory where network weights will be saved.')
args = parser.parse_args()


baseInput = os.path.split(args.input)[1]
outfile = os.path.join(args.outdir, baseInput)

with h5py.File(outfile) as hfOut:
    hfOut.create_group('boardStates')
    hfOut.create_group('labels')

with h5py.File(args.input) as hfIn:
    bigBoard = []
    bigLab = []
    currLen = 0
    for keyCount, mykey in enumerate(hfIn['labels'].keys()):
        tmpBoard = np.array(hfIn['boardStates'][mykey][()], dtype='bool')
        tmpLab = np.array(hfIn['labels'][mykey][()], dtype=np.float32)
        currLen += len(tmpLab)
        bigBoard.append(tmpBoard)
        bigLab.append(tmpLab)

        if currLen > 5e6:
            bigBoard = np.concatenate(bigBoard).astype('bool')
            bigLab = np.concatenate(bigLab).astype(np.float32) 

            print('Saving', mykey, 'of size', currLen)

            with h5py.File(outfile) as hfOut:
                hfOut['boardStates'].create_dataset(mykey, data=bigBoard, compression='gzip')
                hfOut['labels'].create_dataset(mykey, data=bigLab, compression='gzip')

            bigBoard = []
            bigLab = []
            currLen = 0 

        if keyCount == len(hfIn['labels'].keys()) - 1:
            bigBoard = np.concatenate(bigBoard).astype('bool')
            bigLab = np.concatenate(bigLab).astype(np.float32) 

            print('Saving', mykey, 'of size', currLen)

            with h5py.File(outfile) as hfOut:
                hfOut['boardStates'].create_dataset(mykey, data=bigBoard, compression='gzip')
                hfOut['labels'].create_dataset(mykey, data=bigLab, compression='gzip')

 
