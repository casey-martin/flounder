'''pos = {#chessboard
       b: [["r","p","-","-","-","-","P","R"],
           ["n","p","-","-","-","-","P","N"],
           ["b","p","-","-","-","-","P","B"],
           ["q","p","-","-","-","-","P","Q"],
           ["k","p","-","-","-","-","P","K"],
           ["b","p","-","-","-","-","P","B"],
           ["n","p","-","-","-","-","P","N"],
           ["r","p","-","-","-","-","P","R"]],

       # castling rights
       c: [true,true,true,true]

       # enpassant [x, y] coordinates
       e: null

       # side to move
       w: true

       # move counts (moves since last pawn capture, total moves)
       m: [0,1]}'''

def makeMatrix(fen): #board.epd()
    outBoard = []  #Final board
    pieces = fen.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        tmpRow = []  #This is the row I make
        for square in row:
            if square.isdigit():
                for i in range(0, int(square)):
                    tmpRow.append('.')
            else:
                tmpRow.append(thing)
        outBoard.append(tmpRow)
    return(outBoard) 

def sfBoard(pos, x, y):
    if x >= 0 and x <=7 and y >= 0 and y <= 7:
        return(pos['b'][x][y])
    return('x')


def colorflip(pos):
    board = [['-'] * 8 for i in range(8)] 
    for x in range(8):
        for y in range(8):
            board[x][y] = pos['b'][x][7-y]
            color = board[x][y].upper() == board[x][y]
            if color:
                board[x][y] = board[x][y].lower()
            else:
                board[x][y] = board[x][y].upper()
    outPos = {'b':board, 'c':[pos['c'][2], pos['c'][3], pos['c'][0], pos['c'][1]],
              'w':not pos['w'], 'm':[pos['m'][0], pos['m'][1]]} 
    if pos['e'] == None:
        outPos['e'] = None
    else:
        outPos['e'] = [pos['e'][0],7-pos['e'][1]]
    return(outPos)


def sfSum(pos, func, *args):
    outSum = 0
    for x in range(8):
        for y in range(8):
            outSum += func(pos, (x, y), *args)
    return(outSum)
