import os
from pynput import keyboard

class Piece:

    def __init__(self, piece: list[list[int]], xaxis: int=0, yaxis: int=0):
        if False:
            while True:
                i = 10
        self.piece = piece
        self.xaxis = xaxis
        self.yaxis = yaxis

def draw_grid(grid):
    if False:
        print('Hello World!')
    '\n    Draw grid\n    Params: grid (list) of list of int\n    '
    for row in grid:
        for item in row:
            if item == 0:
                print('🔲', end=' ')
            else:
                print('🔳', end=' ')
        print()

def create_grid(limit=10):
    if False:
        while True:
            i = 10
    '\n    Create grid\n    Params: limit (int) limit of grid\n    return: grid (list) of list of int\n    '
    grid = []
    for i in range(limit):
        grid.append([0] * 10)
    return grid

def create_piece_l():
    if False:
        print('Hello World!')
    "\n    Create piece L\n    The piece is a dictionary with keys 'piece', 'xaxis', and 'yaxis'\n    'piece' is a list of list of 3 values\n    'xaxis' is an int of cord x\n    'yaxis' is an int of cord y\n\n    return: Piece\n    "
    piece = []
    piece.append([1, 0, 0])
    piece.append([1, 1, 1])
    return Piece(piece=piece)

def rotate_piece(piece: Piece):
    if False:
        for i in range(10):
            print('nop')
    '\n    Rotate piece\n    Params: piece (list) of list of int\n    return: new Piece with piece rotated 90 degrees clockwise\n    '
    cord_x = piece.xaxis
    cord_y = piece.yaxis
    piece = piece.piece
    max_size = max_limit_internal_x(piece)
    new_piece = []
    if cord_y + max_size > main_limit:
        return Piece(piece=piece, xaxis=cord_x, yaxis=cord_y)
    for i in range(max_size):
        new_piece.append([])
        for j in range(len(piece)):
            new_piece[i].append(piece[len(piece) - j - 1][i])
    return Piece(piece=new_piece, xaxis=cord_x, yaxis=cord_y)

def move(piece: Piece, action='rotate', limit=10):
    if False:
        return 10
    '\n    Move piece\n    Params: piece (list) of list of int\n    Params: action (string) action to move like left, right, down, up\n    Params: limit (int) limit of grid\n    '
    cord_x = piece.xaxis
    cord_y = piece.yaxis
    piece = piece.piece
    internal_max_size_x = max_limit_internal_x(piece)
    internal_max_size_y = max_limit_internal_y(piece)
    if cord_y + internal_max_size_y > limit - 1:
        return Piece(piece=piece, xaxis=cord_x, yaxis=cord_y)
    if action == 'left' and cord_x > 0:
        cord_x -= 1
    elif action == 'right' and cord_x + internal_max_size_x < limit:
        cord_x += 1
    elif action == 'down' and cord_y + internal_max_size_y < limit:
        cord_y += 1
    elif action == 'up' and cord_y > 0:
        cord_y -= 1
    if cord_x < 0:
        cord_x = 0
    if cord_y < 0:
        cord_y = 0
    return Piece(piece=piece, xaxis=cord_x, yaxis=cord_y)

def max_limit_internal_x(piece: list[list[int]]):
    if False:
        print('Hello World!')
    size = 0
    for i in range(len(piece)):
        if len(piece[i]) > size:
            size = len(piece[i])
    return size

def max_limit_internal_y(piece: list[list[int]]):
    if False:
        i = 10
        return i + 15
    return len(piece)

def draw(piece: Piece):
    if False:
        print('Hello World!')
    ' \n    build grid list with values of int 0 or 1\n    '
    grid = create_grid(limit=main_limit)
    cord_x = piece.xaxis
    cord_y = piece.yaxis
    piece = piece.piece
    for i in range(len(piece)):
        for j in range(len(piece[i])):
            if piece[i][j] == 1:
                position_x = cord_x + j
                position_y = cord_y + i
                grid[position_y][position_x] = 1
    draw_grid(grid)

def key_allowed(key):
    if False:
        i = 10
        return i + 15
    if key == keyboard.Key.left:
        return 'left'
    elif key == keyboard.Key.right:
        return 'right'
    elif key == keyboard.Key.down:
        return 'down'
    elif key == keyboard.Key.up:
        return 'up'
    elif key == keyboard.Key.space:
        return 'rotate'
    else:
        return None

def clear_console():
    if False:
        for i in range(10):
            print('nop')
    os.system('cls' if os.name == 'nt' else 'clear')

def on_press(key):
    if False:
        while True:
            i = 10
    global piece
    try:
        action = key_allowed(key)
        if action != None:
            if action == 'rotate':
                piece = rotate_piece(piece)
            else:
                piece = move(piece, action, main_limit)
            clear_console()
            draw(piece)
    except AttributeError:
        pass
main_limit = 10
piece = create_piece_l()
clear_console()
draw(piece)
listener = keyboard.Listener(on_press=on_press)
listener.start()
listener.join()