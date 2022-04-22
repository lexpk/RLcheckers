from abc import ABC, abstractmethod
from time import sleep
import PySimpleGUI as sg
import multiprocessing as mp
from numpy.random import default_rng

def visualize(position):
        layout = [
            [
                sg.Image(
                    Game.IMAGE[3 + position[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0] , subsample=5, key=(i, j)
                ) for j in range(8)
            ]
            for i in range(7, -1, -1)
        ]   
        sg.Window('Checkers', layout, size=(864, 832)).read(close=True)

def _windowloop(window, receiver):
    while True:
        event, _ = window.read(timeout=10)
        if event != '__TIMEOUT__':
            return
        if receiver.poll():
            position = receiver.recv()
            for i in range(7, -1, -1):
                for j in range(8):
                    window[(i, j)].update(
                        Game.IMAGE[3 + position[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0],
                        subsample=5
                    )

class Game():

    IMAGE = {}
    IMAGE[0] = ".\\images\\lightsquare_empty.png"
    IMAGE[1] = ".\\images\\darksquare_light_king.png"
    IMAGE[2] = ".\\images\\darksquare_light_man.png"
    IMAGE[3] = ".\\images\\darksquare_empty.png"
    IMAGE[4] = ".\\images\\darksquare_dark_man.png"
    IMAGE[5] = ".\\images\\darksquare_dark_king.png"


    def __init__(self, position = [1 for _ in range(12)] + [0 for _ in range(8)] + [-1 for _ in range(12)], color = 1, ply = 0):
        self.position = position
        self.color = color
        self.ply = ply
        self.result = None
        self.trace = []
        self.rendering = False

    def _legal_dark_single_captures_no_promotion(piece, position):
        result = []
        if position[piece] > 0:
            if piece < 24:
                if piece % 4 != 0 and position[piece + 3 + (piece//4)%2] in [-1, -2] and position[piece + 7] == 0:
                    result.append(([0 if i in [piece, piece + 3 + (piece//4)%2] else position[piece] if i == piece + 7 else position[i] for i in range(32)], piece + 7))
                if piece % 4 != 3 and position[piece + 4 + (piece//4)%2] in [-1, -2] and position[piece + 9] == 0:
                    result.append(([0 if i in [piece, piece + 4 + (piece//4)%2] else position[piece] if i == piece + 9 else position[i] for i in range(32)], piece + 9))
            if position[piece] == 2 and piece >= 8:
                if piece % 4 != 0 and position[piece - 5 + (piece//4)%2] in [-1, -2] and position[piece - 9] == 0:
                    result.append(([0 if i in [piece, piece - 5 + (piece//4)%2] else position[piece] if i == piece - 9 else position[i] for i in range(32)], piece - 9))
                if piece % 4 != 3 and position[piece - 4 + (piece//4)%2] in [-1, -2] and position[piece - 7] == 0:
                    result.append(([0 if i in [piece, piece - 4 + (piece//4)%2] else position[piece] if i == piece - 7 else position[i] for i in range(32)], piece - 7))
        return result

    def _legal_dark_captures_no_promotion(single_captures):
        captures = single_captures
        for pos, p in captures:
            captures += Game._legal_dark_single_captures_no_promotion(p, pos)
        return [pos for pos, _ in captures]

    def legal_dark_captures_no_promotion(position):
        result = []
        for piece in range(32):
            result += Game._legal_dark_captures_no_promotion(piece, Game._legal_dark_single_captures_no_promotion(piece, position))
        return result

    def legal_dark_non_captures_no_promotion(position):
        result = []
        for piece in range(32):
            if position[piece] > 0:
                if piece < 28:
                    if piece % 8 != 0 and position[piece + 3 + (piece//4)%2] == 0:
                        result.append([0 if i == piece else position[piece] if i == piece + 3 + (piece//4)%2 else position[i] for i in range(32)])
                    if piece % 8 != 7 and position[piece + 4 + (piece//4)%2] == 0:
                        result.append([0 if i == piece else position[piece] if i == piece + 4 + (piece//4)%2 else position[i] for i in range(32)])
                if position[piece] == 2 and piece >= 4:
                    if piece % 8 != 0 and position[piece - 5 + (piece//4)%2] == 0:
                        result.append([0 if i == piece else 2 if i == piece - 5 + (piece//4)%2 else position[i] for i in range(32)])
                    if piece % 8 != 7 and position[piece - 4 + (piece//4)%2] == 0:
                        result.append([0 if i == piece else 2 if i == piece - 4 + (piece//4)%2 else position[i] for i in range(32)])
        return result

    def legal_moves(position, color):
        if color == -1:
            positions = [Game.flip(pos) for pos in Game.legal_moves(Game.flip(position), 1)]
            return positions
        single_captures = []
        for piece in range(32):
            single_captures += Game._legal_dark_single_captures_no_promotion(piece, position)
        if single_captures:
            return [[2 if i >= 28 and capture[i] == 1 else capture[i] for i in range(32)] for capture in Game._legal_dark_captures_no_promotion(single_captures)]
        return [[2 if i >= 28 and capture[i] == 1 else capture[i] for i in range(32)] for capture in Game.legal_dark_non_captures_no_promotion(position)]    

    def render(self):
        self.sender, receiver = mp.Pipe()
        layout = [
            [
                sg.Image(
                    Game.IMAGE[3 + self.position[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0] , subsample=5, key=(i, j)
                ) for j in range(8)
            ]
            for i in range(7, -1, -1)
        ]   
        window = sg.Window('Checkers', layout, size=(864, 832))
        self.windowprocess =  mp.Process(target = _windowloop, args=[window, receiver])
        self.windowprocess.start()
        self.rendering = True          

    def next(self, player, time : float):
        next = player.move(self, time)
        assert next in Game.legal_moves(self.position, self.color)
        self.ply += 1
        self.trace += (self.position, self.color)
        self.position = next
        self.color = -self.color
        if not Game.legal_moves(self.position, self.color):
            self.result = -self.color
        if (self.position, self.color) in self.trace:
            self.result = 0

    def simulate(self, p1, p2 , movetime = 1, rendering = True):
        if rendering:
            self.render()
        while self.result == None:
            if self.color == 1:
                self.next(p1, time = movetime)
            else :
                self.next(p2, time = movetime)
            if rendering:    
                self.sender.send(self.position)

    def flip(position):
        return [-piece for piece in position[::-1]]
            

class Player(ABC):
    @abstractmethod
    def move(self, game : Game, time : float):
        pass

class RandomPlayer(Player):
    def __init__(self):
        self.rng = default_rng()

    def move(self, game : Game, time : float):
        sleep(time)
        return list(self.rng.choice(game.legal_moves()))
