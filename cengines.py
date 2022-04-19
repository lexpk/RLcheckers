import ctypes
import math
from numpy.random import default_rng
import torch
import re
from tqdm import tqdm

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

def tensor_to_c_index(i : int):
    return 6 + (i//4)%2 - 2 * (i % 4), i // 4

def c_to_tensor_index(i : int, j : int):
    assert (i + j)%2, "This square does not occurr in checkers"
    return 4 * j + i//8

def tensor_to_c_position(tensor_board : torch.tensor):
    c_board = [[0 for _ in range(8)] for _ in range(8)]
    for dark_man in tensor_board[0].argwhere():
        i, j = tensor_to_c_index(int(dark_man))
        c_board[i][j] = 6
    for dark_king in tensor_board[1].argwhere():
        i, j = tensor_to_c_index(int(dark_king))
        c_board[i][j] = 10
    for light_man in tensor_board[3].argwhere():
        i, j = tensor_to_c_index(int(light_man))
        c_board[i][j] = 5
    for light_king in tensor_board[4].argwhere():
        i, j = tensor_to_c_index(int(light_king))
        c_board[i][j]
    c_board = [(ctypes.c_int * 8)(*row) for row in c_board]
    return ((ctypes.c_int * 8) * 8)(*c_board)

def get_move_and_evaluation(position : torch.tensor, timelimit = 0.5):
    result = (ctypes.c_char * 1024)()
    engines["simplech"].getmove(
        tensor_to_c_position(position),
        (ctypes.c_int)(2),
        (ctypes.c_double)(timelimit),
        result,
        ctypes.pointer(ctypes.c_int(0)),
        (ctypes.c_int)(),
        (ctypes.c_int)(),
        ctypes.pointer(CBmove()),
    )
    numbers = list(map(int, re.findall(r'\-?\d+', str(result[:100]))))
    return numbers[0], abs(numbers[1]), 1./(1. + math.exp(float(-min(max(numbers[5], -500), 500))/50.))

def random_positions(size : int, time_per_pos : float = 0.5, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    data = []
    rng = default_rng()
    for i in tqdm(range(0, size, 2)):
        kings_prob = 0.3*rng.random()
        light_man_cnt = rng.integers(1, 13)
        light_king_cnt = rng.binomial(12 - light_man_cnt, kings_prob)
        dark_man_cnt = 1 + rng.binomial(11, (float)(light_man_cnt - 1)/12.)
        dark_king_cnt = rng.binomial(12 - dark_man_cnt, kings_prob)
        available_squares = list(range(32))
        light_man_pos = [False for _ in range(32)]
        light_king_pos = [False for _ in range(32)]
        dark_man_pos = [False for _ in range(32)]
        dark_king_pos = [False for _ in range(32)]
        for cnt, pos, cond in zip(
            [light_man_cnt, light_king_cnt, dark_man_cnt, dark_king_cnt],
            [light_man_pos, light_king_pos, dark_man_pos, dark_king_pos],
            [lambda x : x < 28, lambda _ : True, lambda x : x >= 4, lambda _ : True]
        ):
            while cnt:
                s = rng.choice(available_squares)
                if cond(s):
                    pos[s] = True
                    available_squares.remove(s)
                    cnt -= 1
        data.append([
                light_man_pos,
                light_king_pos,
                [x or y for x, y in zip(light_man_pos, light_king_pos)],
                dark_man_pos,
                dark_king_pos,
                [x or y for x, y in zip(dark_man_pos, dark_king_pos)]
            ])
        mirrored_light_man_pos = [light_man_pos[4*(i//4) + (3 - i%4)] for i in range(32)]
        mirrored_light_king_pos = [light_king_pos[4*(i//4) + (3 - i%4)] for i in range(32)]
        mirrored_dark_man_pos = [dark_man_pos[4*(i//4) + (3 - i%4)] for i in range(32)]
        mirrored_dark_king_pos = [dark_king_pos[4*(i//4) + (3 - i%4)] for i in range(32)]
        data.append([
                mirrored_light_man_pos,
                mirrored_light_king_pos,
                [x or y for x, y in zip(mirrored_light_man_pos, mirrored_light_king_pos)],
                mirrored_dark_man_pos,
                mirrored_dark_king_pos,
                [x or y for x, y in zip(mirrored_dark_man_pos, mirrored_dark_king_pos)]
            ])
    return data
        
        