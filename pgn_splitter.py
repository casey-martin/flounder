import argparse
import tempfile
import os
import re


pattern = re.compile('Event')

def flatten(l):
    flat = [val for sublist in l for val in sublist]
    return(flat)

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return(out)


def numLines(myfile):

    return(sum(1 for line in open(myfile)))


def getGames(pgn):
    '''PGN files begin with the string "[Event". Return indices where a each
       game begins.'''

    hits = []
    for i, line in enumerate(open(pgn)):
        if line[1:6] == 'Event':
            hits.append(i) 

    return(hits)


def getJobIndices(pgn, job, nprocs):
    '''Subdivides the pgn into n equal chunks (nprocs) with roughly an 
       equal number of games. Returns the indices corresponding to the intervals
       for the job. '''

    hits = getGames(pgn)

    chunked = chunk(hits, nprocs)

    startChunks = [i[0] for i in chunked]
    stopChunks = startChunks[1:] + [numLines(pgn)]

    indices = [(i,j) for i,j in zip(startChunks, stopChunks)]

    return(indices[job])


def writePgnChunk(pgn, job, nprocs, outfile='pgn_chunk.pgn'): 
    '''Writes a new file with 1/nprocs the number of games in the original
       pgn.'''

    pgnBasename = os.path.basename(pgn)
    pgnBasename = os.path.splitext(pgnBasename)[0]

    jobIndices = getJobIndices(pgn, job, nprocs)
    
    with open(outfile, 'w+') as outfile:
        for i, line in enumerate(open(pgn)):
            if jobIndices[0] <= i < jobIndices[1]:
                outfile.write(line)
            if i >= jobIndices[1]:
                break


'''parser = argparse.ArgumentParser()
parser.add_argument('--pgn', type=str, help='Input pgn file to split.')
parser.add_argument('--outdir', type=str, help='Directory to which the file will be written.')
parser.add_argument('--nprocs', type=int, help='Number of processors to use.')
parser.add_argument('--job', type=int, help='Job id, using 0 based numbering.')

args = parser.parse_args()

# Identify lines where a new game begins. 

hits = getGames(args.pgn)
jobIndices = getJobIndices(hits)

pgnBasename = os.path.basename(args.pgn)
pgnBasename = os.path.splitext(pgnBasename)[0]'''





 
            
