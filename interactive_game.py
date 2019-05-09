import argparse
from copy import deepcopy
from io import StringIO  # Python3
from tensorflow import keras
from tensorflow.python.keras import layers
import chess
import numpy as np
import sys
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to ANN evaluator weights.')
parser.add_argument('--playerWhite', type=bool, default = True, help='True/False. Will the player play as white?')
args = parser.parse_args()


def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(2048, activation='relu', input_shape=(769,)))
    model.add(layers.Dense(2048, activation=tf.nn.relu))
    model.add(layers.Dense(2048, activation=tf.nn.relu))
    model.add(tf.layers.Flatten())
    model.add(layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.7, decay=1e-08, nesterov=True)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return(model)

class greedySearch:
    def __init__(self, model, board):
        self.board = board
        self.model = model
        self.piece_id = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
        self.pieceIdDict = {'p':-1,'P':1,
                           'r':-2,'R':2,
                           'n':-3,'N':3,
                           'b':-4,'B':4,
                           'q':-5,'Q':5,
                           'k':-6,'K':6}
    def board2Vec(self, board):
        '''Given a chess.Board() object, return a vector representation
           of the board's current state as 1x65 vector, where the 65th
           position indicates whose turn it is.'''
        
        piece_map = board.piece_map()
        stateVec = np.zeros(65)
        for i in piece_map:
            mypiece = str(piece_map[i])
            stateVec[i] = self.pieceIdDict[mypiece]

        # gives whose turn it is. 1 == white, 0 == black
        stateVec[64] = int(board.turn)
            
        return(stateVec)

    def stateVec2Mat(self, stateVec):
        board = stateVec[:64]
        whiteTurn = stateVec[64]
        
        outMatrix = []
        for i in self.piece_id:
            zslice = board == i
            outMatrix.append(zslice.astype(int))
        outMatrix.append(np.array([whiteTurn]))
        
        return(np.hstack(outMatrix))

    def board2Mat(self, board):
        return(self.stateVec2Mat(self.board2Vec(board)))

    def bestMove(self):
        ann_inputs = []
        myLegalMoves = []
        for move in self.board.legal_moves:
            myLegalMoves.append(move)
            tmpBoard = deepcopy(self.board)
            tmpBoard.push(move)
            ann_inputs.append(self.board2Mat(tmpBoard))
        evals = self.model.predict(np.array(ann_inputs))
        topHits = np.hstack(np.where(evals == np.amin(evals)))
        move = np.random.choice(topHits, 1)[0]
        return(myLegalMoves[move])
        #return(topHits[0])


def ply(board, model, playerWhite = True):
    computerColor = not playerWhite
    
    if board.result() == '*':    
        if playerWhite:
            while True:
                try:
                    move = str(input('What is your move? '))
                    if move == 'help':
                        print(board.legal_moves)
                    else:
                        board.push_san(move)
                except ValueError:
                    print('Error: invalid move. Try again')
                    continue
                except KeyboardInterrupt:
                    break
                else:
                    break
            print(board)
            print('\n')
            bestAction = greedySearch(model, board).bestMove() 
            board.push(bestAction)
            print(board)
            
        else:
            bestAction = greedySearch(model, board).bestMove() 
            board.push(bestAction)
            print(board)
            print('\n')
            while True:
                try:
                    move = str(input('What is your move?'))
                    if move == 'help':
                        print(board.legal_moves)
                    else:
                        board.push_san(move)
                except ValueError:
                    print('Error: invalid move. Try again')
                    continue
                except KeyboardInterrupt:
                     break
                else:
                    break
            print(board)

def game(board, model, playerWhite=True):
    while board.result() == '*':
        ply(board,model, playerWhite)

def main():
    model = build_model()
    model.load_weights(args.model)
    board = chess.Board()
    game(board, model, args.playerWhite)

if __name__ == '__main__':
    main()


