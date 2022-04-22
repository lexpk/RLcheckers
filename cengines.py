import ctypes
from numpy.random import default_rng
import torch
import re
from tqdm import tqdm
from checkers import Game, Player
from math import exp

# Engines used in Checkerboard (http://www.fierz.ch/checkers.htm)
engines = {name : ctypes.WinDLL(f".\\engines\\{name}64.dll") for name in ["easych", "simplech"]}

class coor(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]

class CBmove(ctypes.Structure):
    _fields_ = [
        ("jumps", ctypes.c_int),
        ("newpiece", ctypes.c_int),
        ("oldpiece", ctypes.c_int),
        ("from", coor),
        ("to", coor),
        ("path", coor * 12),
        ("del", coor * 12),
        ("delpiece", ctypes.c_int * 12)
    ]

for engine in engines.values():
    engine.getmove.argtypes = [
        (ctypes.c_int * 8) * 8,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_char * 1024,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(CBmove),
    ]

def index_32_to_64(i : int):
    return (i//4)%2 + 2 * (i % 4), i // 4

def index_64_to_32(i : int, j : int):
    assert not (i + j)%2, "This square does not occurr in checkers"
    return 4 * j + i//2

def position_32_to_64(board):
    c_board = [[0 for _ in range(8)] for _ in range(8)]
    for ind, piece in enumerate(board):
        if piece == 1:
            i, j = index_32_to_64(ind)
            c_board[i][j] = 6
        if piece == 2:
            i, j = index_32_to_64(ind)
            c_board[i][j] = 10
        if piece == -1:
            i, j = index_32_to_64(ind)
            c_board[i][j] = 5
        if piece == -2:
            i, j = index_32_to_64(ind)
            c_board[i][j] = 9 
    c_board = [(ctypes.c_int * 8)(*row) for row in c_board]
    return ((ctypes.c_int * 8) * 8)(*c_board)

def get_output(position, color, timelimit = 0.5, name = "simplech"):
    result = (ctypes.c_char * 1024)()
    engines[name].getmove(
        position_32_to_64(position),
        (ctypes.c_int)(2 if color == 1 else 1),
        (ctypes.c_double)(timelimit),
        result,
        ctypes.pointer(ctypes.c_int(0)),
        (ctypes.c_int)(),
        (ctypes.c_int)(),
        ctypes.pointer(CBmove()),
    )
    numbers = list(map(int, re.findall(r'\-?\d+', str(result[:100]))))
    return numbers[0], abs(numbers[1]), 1./(1. + exp(float(-min(max(numbers[5], -500), 500))/50.))

class Easy(Player):
    def __init__(self):
        self.next = None

    def move(self, game: Game, time: float):
        moves = Game.legal_moves(game.position, game.color)
        if len(moves) == 1:
            return moves[0]
        else:
            fr, to, _ = get_output(game.position, game.color, timelimit=time, name="easych")
            fr, to = fr - 1, to - 1
            fr, to = 4*(fr//4) + 3 - (fr%4), 4*(to//4) + 3 - (to%4)
            if fr not in range(32) or to not in range(32):
                return moves[-1]
            return next(filter(lambda move : move[fr] != game.position[fr] and move[to] != game.position[to], moves))

class Simple(Player):
    def __init__(self):
        self.next = None

    def move(self, game: Game, time: float):
        moves = Game.legal_moves(game.position, game.color)
        if len(moves) == 1:
            return moves[0]
        else:
            fr, to, _ = get_output(game.position, game.color, timelimit=time, name="simplech")
            fr, to = fr - 1, to - 1
            fr, to = 4*(fr//4) + 3 - (fr%4), 4*(to//4) + 3 - (to%4)
            if fr not in range(32) or to not in range(32):
                return moves[-1]
            return next(filter(lambda move : move[fr] != game.position[fr] and move[to] != game.position[to], moves))


        