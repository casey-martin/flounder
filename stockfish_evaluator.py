from chess.engine import Cp
from copy import deepcopy
from pgn_splitter import writePgnChunk, getGames
import argparse
import chess
import chess.engine
import chess.pgn
import numpy as np
import os
import sys
import time
import tempfile

# Lernaecera lumpi
pieceIdDict = {'p':-1,'P':1,
               'r':-2,'R':2,
               'n':-3,'N':3,
               'b':-4,'B':4,
               'q':-5,'Q':5,
               'k':-6,'K':6}

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
    
    
def board2Vec(board):
    '''Given a chess.Board() object, return a vector representation
       of the board's current state as 1x65 vector, where the 65th
       position indicates whose turn it is.'''
    
    piece_map = board.piece_map()
    myarr = np.zeros(65)
    for i in piece_map:
        mypiece = str(piece_map[i])
        myarr[i] = pieceIdDict[mypiece]

    # gives whose turn it is. 1 == white, 0 == black
    myarr[64] = int(board.turn)
        
    return(myarr)

# returns Stockfish's evaluation for a given position.
# Checkmate's value is 10000. Mate in n is given as 
# 10000 - n. 
def sfScore(board, myengine, limit):
    myanalysis = myengine.analyse(board, limit)
    myscore = myanalysis.get('score').relative.score(mate_score=10000)
        
    return(myscore)
        

def analyzeGame(mygame, myengine, limit):
    '''Iterates through mainline in the game and evaluates each position.
       Returns the board state as a vector its corresponding evaluation score.
       Evaluation is given in centipawns'''

    moveScores = []
    boardStates = []
    board = mygame.board()
    for move in mygame.mainline_moves():
        board.push(move)
        myscore = sfScore(board,myengine,limit)
        mystate = board2Vec(board)
        moveScores.append(myscore)
        boardStates.append(mystate)
   
    boardStates = [list(i) for i in boardStates] 
    for i,j in zip(boardStates, moveScores):
        i.append(j)

    cleanBoardStates = []
    for i in boardStates:
        cleanBoardStates.append(list(map(int, i)))

 
    return(cleanBoardStates)


def analyzeGameVar(mygame, myengine, limit):
    '''Iterates through mainline in the game and evaluates each position.
       Returns the board state as a vector its corresponding evaluation score.
       Evaluation is given in centipawns'''

    moveScores = []
    boardStates = []
    board = mygame.board()
    gameLen = len(list(mygame.mainline_moves()))
    for moveNum, move in enumerate(mygame.mainline_moves()):
        board.push(move)
        if moveNum < gameLen:
            for possMove in board.legal_moves:
                tmpBoard = deepcopy(board)
                
                # generate and evaluate all legal moves
                tmpBoard.push(possMove)
                myscore = sfScore(tmpBoard,myengine,limit)
                mystate = board2Vec(tmpBoard)
                moveScores.append(myscore)
                boardStates.append(mystate)

                # generate and evaluate best response to all legal moves
                bestMove = myengine.play(tmpBoard, mylimit)
                tmpBoard.push(bestMove.move)
                myscore = sfScore(tmpBoard,myengine,limit)
                mystate = board2Vec(tmpBoard)
                moveScores.append(myscore)
                boardStates.append(mystate)
        else:
            myscore = sfScore(board,myengine,limit)
            mystate = board2Vec(board)
            moveScores.append(myscore)
            boardStates.append(mystate)

   
    boardStates = [list(i) for i in boardStates] 
    for i,j in zip(boardStates, moveScores):
        i.append(j)

    cleanBoardStates = []
    for i in boardStates:
        cleanBoardStates.append(list(map(int, i)))

 
    return(cleanBoardStates)



def scorePgnChunk(pgn, myengine, limit, job, nprocs, outdir, verbose=True):
    
    
    # creates a shard of the original pgn.
    with tempfile.TemporaryDirectory(prefix='pgn_batch') as temp:
        pgnBasename = os.path.basename(pgn)
        pgnBasename = os.path.splitext(pgnBasename)[0]

        outname = pgnBasename + '_' + 'scored' + '_' + str(job) + '.csv'
        outname = os.path.join(outdir, outname)    
        #chop up input pgn and write to tmp file. 
        pgnChunk = os.path.join(temp, str(job))
        writePgnChunk(pgn, job, nprocs, outfile = pgnChunk)

        with open(pgnChunk) as myPgnShard, open(outname, 'w+') as outfile:
            # iterate over each game in the pgn shard and score each move.
            numGames = len(getGames(pgnChunk))

            if verbose:
                # updates the time 
                durations = []
                for i in range(numGames):
                    start = time.time() 

                    mygame = chess.pgn.read_game(myPgnShard)
                    boardStates = analyzeGameVar(mygame, myengine, limit)
                    #print(boardStates)
                    
                    for state in boardStates:
                        outfile.write(','.join(map(str,state)) + '\n')
                    
                    stop = time.time()
                    elapsed = round((stop - start)/3600, 3)
                    durations.append(elapsed)
                    timeRemaining = round(np.mean(durations) * (numGames - i + 1), 2)
                    
                    # print and update estimated time remaining.
                    print('Job:', job, round(((i+1)/numGames)*100, 2), "% percent complete.", timeRemaining,
                          'hours remaining.', end='\r' )
                    sys.stdout.flush()

            else:
                for i in range(numGames):
                    mygame = chess.pgn.read_game(myPgnShard)
                    boardStates = analyzeGameVar(mygame, myengine, limit)
                    
                    for state in boardStates:
                        outfile.write(','.join(map(str,state)) + '\n')

    print('\nJob:', job, 'complete.')

 
parser = argparse.ArgumentParser()
parser.add_argument('--pgn', type=str, help='input pgn file to split.')
parser.add_argument('--outdir', type=str, help='Directory to which the file will be written.')
parser.add_argument('--job', type=int, help='Job ID given as integer.')
parser.add_argument('--nprocs', type=int, help='Total number of jobs.')
parser.add_argument('--verbose', type=bool, default=True,
                    help='Show job status and estimated time to completion.')
parser.add_argument('--timelimit', type=float, default=0.1, help='Wall time for move analysis.')
parser.add_argument('--engine', type=str, default='/usr/bin/stockfish/', help='Path to engine.')
args = parser.parse_args()

def main():
    # set up engine parameters. 
    myengine = chess.engine.SimpleEngine.popen_uci(args.engine)
    limit = chess.engine.Limit(time=args.timelimit)


    scorePgnChunk(pgn=args.pgn, myengine=myengine, limit=limit, 
                 job=args.job, nprocs=args.nprocs, outdir=args.outdir,
                 verbose=args.verbose)
    myengine.quit()
    return()


if __name__ == '__main__':
    main()
