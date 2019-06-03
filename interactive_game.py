import argparse
from collections import OrderedDict
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
parser.add_argument('--model_1', type=str, help='Path to ANN evaluator weights.')
parser.add_argument('--model_2', type=str, help='Path to ANN evaluator weights.')
parser.add_argument('--playerWhite', type=bool, default = True, help='True/False. Will the player play as white?')
args = parser.parse_args()

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
        #topHits = np.hstack(np.where(evals == evals.min()))
        #move = np.random.choice(topHits, 1)[0]
        move = np.argmin(evals)
        print('Top move:', myLegalMoves[move])
        #print(OrderedDict(zip([str(i) for i in myLegalMoves], [float(j) for j in evals])))
        return(myLegalMoves[move], evals.min())
        
        #return(topHits[0])

    def currentEval(self):
        return(self.model.predict(np.array( [self.board2Mat(self.board)] )))

class lookAhead:
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


    def alphabeta(self, position, depth=3,alpha=-1000000,beta=1000000):
        node = deepcopy(position)
        if depth == 0:
            return(self.model.predict(np.array( [self.board2Mat(node)] )))
        
        #minimizing
        if depth % 2:
            minEval = 1000000
            myLegalMoves = []
            for move in position.legal_moves:
                child = deepcopy(node)
                child.push(move)
                childEval = self.alphabeta(child, depth-1)
                minEval = min(minEval, childEval)
                beta = min(beta, childEval)
                if beta <= alpha:
                    break
            return(minEval)

        else:
            maxEval = -1000000
            myLegalMoves = []
            for move in position.legal_moves:
                child = deepcopy(node)
                child.push(move)
                childEval = self.alphabeta(child, depth-1)
                maxEval = max(maxEval, childEval)
                alpha = max(alpha, childEval)
                if beta <= alpha:
                    break
            return(maxEval)


    def minimax(self, position, depth=3):
        node = deepcopy(position)
        if depth == 0:
            return(self.model.predict(np.array( [self.board2Mat(node)] )))
        
        #minimizing
        if depth % 2:
            minEval = 1000000
            myLegalMoves = []
            for move in position.legal_moves:
                child = deepcopy(node)
                child.push(move)
                childEval = self.minimax(child, depth-1)
                minEval = min(minEval, childEval)
            return(minEval)

        else:
            maxEval = -1000000
            myLegalMoves = []
            for move in position.legal_moves:
                child = deepcopy(node)
                child.push(move)
                childEval = self.minimax(child, depth-1)
                maxEval = max(maxEval, childEval)
            return(maxEval)



    def bestMoveMM(self, depth=3):
        evals = []
        myLegalMoves = []
        for move in self.board.legal_moves:
            myLegalMoves.append(move)
            tmpBoard = deepcopy(self.board)
            tmpBoard.push(move)
            evals.append(self.minimax(tmpBoard, depth=depth))

        #topHits = np.hstack(np.where(evals == evals.min()))
        #move = np.random.choice(topHits, 1)[0]
        move = np.argmin(evals)
        print('Top move:', myLegalMoves[move])
        #print(OrderedDict(zip([str(i) for i in myLegalMoves], [float(j) for j in evals])))
        return(myLegalMoves[move], min(evals))
        
        #return(topHits[0])

    def bestMoveAB(self, depth=3):
            evals = []
            myLegalMoves = []
            for move in self.board.legal_moves:
                myLegalMoves.append(move)
                tmpBoard = deepcopy(self.board)
                tmpBoard.push(move)
                evals.append(self.alphabeta(tmpBoard, depth=depth))

            #topHits = np.hstack(np.where(evals == evals.min()))
            #move = np.random.choice(topHits, 1)[0]
            move = np.argmin(evals)
            print('Top move:', myLegalMoves[move])
            #print(OrderedDict(zip([str(i) for i in myLegalMoves], [float(j) for j in evals])))
            return(myLegalMoves[move], min(evals))


    #    def currentEval(self):
#        return(self.model.predict(np.array( [self.board2Mat(self.board)] )))






def ply(board, model_1, model_2):
    

    while True:
        advance = input()
        if advance != None:
            print('\n')
            print('White turn')
            #print('Current evaluation:', greedySearch(model_1, board).currentEval())
            searchStrat = lookAhead(model_1, board)
            t0=time.time()
            bestAction, actionEval = searchStrat.bestMoveAB(depth=2) 
            t1 = time.time()
            print('Time to move:', t1-t0)
            print('Move eval:',1-actionEval)
            board.push(bestAction)
            print(board)
        advance = input()
        if advance != None:
            print('\n')
            print('Black turn')
            searchStrat = lookAhead(model_2, board)
            t0 = time.time()
            bestAction, actionEval = searchStrat.bestMoveMM(depth=2) 
            t1 = time.time()
            print('Time to move:',t1-t0)
            print('Move eval:', 1-actionEval)
            board.push(bestAction)
            print(board)
            
    
def game(board, model_1, model_2):
    while board.result() == '*':
        ply(board,model_1, model_2)

def main():
    model_1 = build_model()
    model_1.load_weights(args.model_1)
    model_2 = build_model()
    model_2.load_weights(args.model_2)
    board = chess.Board()
    game(board, model_1, model_2)

if __name__ == '__main__':
    main()


