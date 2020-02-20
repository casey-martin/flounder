
from utils.common import board2Vec
from copy import deepcopy
from datetime import datetime
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import argparse
import h5py
import chess
import chess.engine
import numpy as np
import os
import sys
import tensorflow as tf
import time

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
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))

    model.add(layers.Dense(1858, activation='relu'))
    optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.7, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer])
    return(model)


class greedySearch:
    def __init__(self, model, board):
        self.board = board
        self.model = model
    
    def bestMove(self):
        ann_inputs = []
        myLegalMoves = []
        for move in self.board.legal_moves:
            myLegalMoves.append(move)
            tmpBoard = deepcopy(self.board)
            tmpBoard.push(move)
            ann_inputs.append(board2Vec(tmpBoard))
        evals = self.model.predict(np.array(ann_inputs))

        if self.board.turn:
            move = np.argmax(evals)
            return(myLegalMoves[move])
        else:
            move = np.argmin(evals)
            return(myLegalMoves[move])


def annPly(board, model):
    if board.result() == '*':
        searchStrat = greedySearch(model, board)
        bestAction = searchStrat.bestMove()
        board.push(bestAction)
    else:
        return

def teacherPly(board, engine, limit):
    if board.result() == '*':
        bestAction = engine.play(board, limit)
        board.push(bestAction.move)
    else:
        return

def teacherScore(board, teacher, limit):
    myanalysis = teacher.analyse(board, limit)
    myscore = myanalysis.get('score').white().score(mate_score = 150*100)
    return(myscore/100.0)

def teacherEvalPositions(board, teacher, limit):
    boardStates = []
    labels = []
    for move in board.legal_moves:
        board.push(move)
        boardStates.append(board2Vec(board))
        labels.append(teacherScore(board, teacher, limit))
        board.pop()

    return(boardStates, labels)
        


def game(board, ann, teacher, limit, killGame = 100, dilution=0.5, annWhite=True):
    boardStates = []
    labels = []
    turnCount = 0

    if annWhite:
        while board.result() == '*':
            try:
                tmpBoardStates, tmpLabels = teacherEvalPositions(board, teacher, limit)
                boardStates += tmpBoardStates
                labels += tmpLabels

                annPly(board, ann)
                
                # teacher evals 
                if board.result() == '*':            
                    tmpBoardStates, tmpLabels = teacherEvalPositions(board, teacher, limit)
                    boardStates += tmpBoardStates
                    labels += tmpLabels

                # random teacher move
                if np.random.uniform(0,1) > dilution:
                    legalMoves = [move for move in board.legal_moves]
                    randomMove = np.random.choice(legalMoves)
                    board.push(randomMove)
                else:
                    teacherPly(board, teacher, limit)
     
                turnCount += 1        
                if turnCount > killGame:
                    break
     

            except:
                break

    else:
        while board.result() == '*':
            try:
                tmpBoardStates, tmpLabels = teacherEvalPositions(board, teacher, limit)
                boardStates += tmpBoardStates
                labels += tmpLabels
                if np.random.uniform(0,1) > dilution:
                    legalMoves = [move for move in board.legal_moves]
                    randomMove = np.random.choice(legalMoves)
                    board.push(randomMove)
                else:
                    teacherPly(board, teacher, limit)
                if board.result() == '*':
                    tmpBoardStates, tmpLabels = teacherEvalPositions(board, teacher, limit)
                    boardStates += tmpBoardStates
                    labels += tmpLabels


                annPly(board, ann)
                turnCount += 1        
                if turnCount > killGame:
                    break

            except:
                break

            
    return(boardStates, labels)


def trainRounds(ann, teacher, limit, rounds, dilution):
    gameResults = {'win':0, 'loss':0, 'tie':0}
    boardStates = []
    labels = []


    numMoves = 0
    t1 = time.time()


    for i in range(args.rounds):
        board = chess.Board()
        tmpBoardStates, tmpLabels = game(board, ann, teacher, limit, annWhite=True, dilution=dilution)

        boardStates += tmpBoardStates
        labels += tmpLabels        

        myresult = board.result().split('-')
        if myresult[0] == '1':
            gameResults['win'] += 1
        if myresult[0] == '1/2' or myresult[0] == '*':
            gameResults['tie'] += 1
        else:
            gameResults['loss'] += 1
            
        print(gameResults)

        board = chess.Board()
        tmpBoardStates, tmpLabels = game(board, ann, teacher, limit, annWhite=False, dilution=dilution)
        boardStates += tmpBoardStates
        labels += tmpLabels        

        myresult = str(board.result()).split('-')

        if myresult[0] == '0':
            gameResults['win'] += 1
        if myresult[0] == '1/2' or myresult[0] == '*':
            gameResults['tie'] += 1
        else:
            gameResults['loss'] += 1
 
        print(gameResults)

    return(boardStates, labels, gameResults)


def rescale(cp, maxCp = 150, minCp = -150):
    centipawn_norm = np.clip(cp, minCp, maxCp)
    centipawn_norm = (centipawn_norm - minCp)/(maxCp - minCp)
    return(centipawn_norm)


parser = argparse.ArgumentParser()
parser.add_argument('--ann', type=str, required=True, help='Path to ANN evaluator weights.')
parser.add_argument('--teacher', type=str, default='/usr/bin/stockfish/',  help='Path to conventional chess engine.')
parser.add_argument('--depth_2', type=int, default=5, help='Search depth for conventional chess engine.')
parser.add_argument('--dilution', type=float, default=0.8, help='Relative engine strength. Use 1 for no random moves.')
parser.add_argument('--rounds', type=int, default=80, help='Results in 2N number of games where N is number of rounds.')
parser.add_argument('--iters', type=int, default=10, help='Number of rounds of training')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs during fitting.')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_set', type=str, required=True, help='Path to validation data.')
parser.add_argument('--outdir', type=str, required=True)



args = parser.parse_args()

def main():
    ann = build_model()
    ann.load_weights(args.ann)
    teacher= chess.engine.SimpleEngine.popen_uci(args.teacher)
    limit = chess.engine.Limit(depth=args.depth_2)
    testX = []
    testY = []
    with h5py.File(args.test_set) as hf:
        for i in hf['labels'].keys():
            tmpX = np.array(hf['boardStates'][i][()])
            tmpY = np.array(hf['labels'][i][()])
            
            testX.append(tmpX)
            testY.append(tmpY)

    testX = np.concatenate(testX)
    testY = np.concatenate(testY)
    testY = rescale(testY)

    for j in range(args.iters):
        boardStates, labels, gameResults  = trainRounds(ann, teacher, limit, args.rounds, args.dilution)
        boardStates = np.array(boardStates).astype('bool')
        labels = np.array(labels)
        labels = rescale(labels)
        model_history = ann.fit(x=boardStates, y=labels,
                validation_data = (testX, testY),
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=2,
                shuffle=True)
        ann.save_weights(os.path.join(args.outdir, "model_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S") + '.h5'))

if __name__  == '__main__':
    main()

