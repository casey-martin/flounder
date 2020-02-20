def board2Vec(board):
    import numpy as np
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

