
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, help='Input training data')
parser.add_argument('--outdir', type=str, help='Directory to which file will be written.')
args = parser.parse_args()

scores = []
with open(args.infile) as f:
    for line in f.readlines():
        scores.append(int(line.strip().split(',')[-1]))

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(scores, bins=40)

outName = os.path.basename(args.infile) + '.png'

fig.savefig(os.path.join(args.outdir, outName))
