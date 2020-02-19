from tqdm import tqdm
from policy import move2Vec
import argparse
import chess.pgn
import gc
import h5py
import os.path as path
import numpy as np

def numGames(pgn):
    '''PGN files begin with the string "[Event". Return indices where a each
       game begins.'''
    print('Reading data...')
    if path.getsize(pgn)/1e9 > 0.5:
        print('This may take a while. Make some coffee or take a swoop break.')
    hits = 0 
    with tqdm(total = path.getsize(pgn),ncols=75, dynamic_ncols=True) as pbar:
        with open(pgn) as f:
            for line in f:
                pbar.update(len(line.encode('utf-8')))
                if line[1:6] == 'Event':
                    hits += 1 

    return(hits)


def scrapeGame(game):

    board = game.board()
    gameResult = game.headers['Result'] 
    boardStates = []
    labels = []
    # get labels for white
    if gameResult == '1-0':
        moveCount = 0
        for move in game.mainline_moves():
            if moveCount % 2 == 0:
                boardStates.append(board2Vec(board))
                labels.append(move2Vec(str(move)))
                board.push(move)
            else:
                board.push(move)

    # get labels for black
    elif gameResult == '0-1':
        moveCount = 0
        for move in game.mainline_moves():
            if moveCount % 2 == 1:
                boardStates.append(board2Vec(board))
                labels.append(move2Vec(str(move)))
                board.push(move)
            else:
                board.push(move)

    # get labels for both
    else:
        for move in game.mainline_moves():
            boardStates.append(board2Vec(board))
            labels.append(move2Vec(str(move)))
            board.push(move)

    return(boardStates, labels)
    


def board2Vec(board):
    '''Given a chess.Board() object, return a vector representation
       of the board's current state as 1x261 vector, where indices
       0:256 are the 64 squares of the chess board populated by a nibble
       representing the various pieces. See pieceIdDict for encodings.
       Indices 256:261 give the turn, and castling rights for black and
       white.'''

    pieceIdDict = {'p' : np.array([0,0,0,1], dtype='bool'), 'P' : np.array([0,0,1,0], dtype='bool'),
                   'r' : np.array([0,0,1,1], dtype='bool'), 'R' : np.array([0,1,0,0], dtype='bool'),
                   'n' : np.array([0,1,0,1], dtype='bool'), 'N' : np.array([0,1,1,0], dtype='bool'),
                   'b' : np.array([0,1,1,1], dtype='bool'), 'B' : np.array([1,0,0,0], dtype='bool'),
                   'q' : np.array([1,0,0,1], dtype='bool'), 'Q' : np.array([1,0,1,0], dtype='bool'),
                   'k' : np.array([1,0,1,1], dtype='bool'), 'K' : np.array([1,1,0,0], dtype='bool')}



    piece_map = board.piece_map()
    mybitBoard = []
    for i in range(64):
        try:
            mypiece = str(piece_map[i])
            mybitBoard.append(pieceIdDict[mypiece])
        except KeyError:
            mybitBoard.append(np.array([0,0,0,0], dtype='bool'))

    # gives whose turn it is. 1 == white, 0 == black
    mybitBoard.append(np.array([board.turn], dtype='bool'))

    # append king and queenside castling rights for white and black
    for color in (True, False):
        mybitBoard.append(np.array([board.has_queenside_castling_rights(color)], dtype='bool'))
        mybitBoard.append(np.array([board.has_kingside_castling_rights(color)], dtype='bool'))

    mybitBoard = np.concatenate(mybitBoard)

    return(mybitBoard)


parser = argparse.ArgumentParser()
parser.add_argument('--pgn', type=str, required=True, help='''input pgn for evaluation extraction.''')
parser.add_argument('--outdir', type=str, required=True, help='''destination for output files.''')
parser.add_argument('--chunksize', type=float, default=5e6, help='''number of positions to be output in each chunk.
                                                                     scientific notation is supported.''')
args = parser.parse_args()

def main():
    databaseSize = numGames(args.pgn)
    outName = path.join(args.outdir, path.basename( path.splitext(args.pgn)[0] ) + '.hf')
    with h5py.File(outName) as hf:
       stateGroup = hf.create_group('boardStates')
       labelGroup = hf.create_group('labels')        

    with open(args.pgn) as f:

        totalCount = 0
        evalCount = 0
        boardStates = []
        myevalList = []
        for i in tqdm(range(databaseSize),ncols=75, dynamic_ncols=True):
        
            mygame = chess.pgn.read_game(f)
            myevals = scrapeGame(mygame)
            totalCount += 1

            if myevals is not None:
                evalCount += 1
                boardStates += myevals[0]
                myevalList += myevals[1]

            if len(myevalList) > args.chunksize:
                with h5py.File(outName) as hf:

                    hf['boardStates'].create_dataset('_' + str(i),  data = np.array(boardStates).astype('bool'), compression='gzip')
                    hf['labels'].create_dataset('_' + str(i), data = np.array(myevalList).astype('bool'), compression='gzip')
    
                boardStates = []
                myevalList = []
                gc.collect()
                print('Array written')
            if i == databaseSize-1:
                with h5py.File(outName) as hf:

                    hf['boardStates'].create_dataset('_' + str(i),  data = np.array(boardStates).astype('bool'), compression='gzip')
                    hf['labels'].create_dataset('_' + str(i), data = np.array(myevalList).astype(np.float32), compression='gzip')
    
                boardStates = []
                myevalList = []
                gc.collect()

if __name__ == '__main__':
    main()
