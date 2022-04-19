from abc import ABC, abstractmethod
from typing import final
import PySimpleGUI as sg
import multiprocessing as mp

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


    def __init__(self):
        self.position = [1 for _ in range(12)] + [0 for _ in range(8)] + [-1 for _ in range(12)]
        self.ply = 0
        self.color = 1
        self.rendering = False

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
        

    def is_over(self):
        if 1 not in self.position and 2 not in self.position:
            return True, -1
        if -1 not in self.position or -2 not in self.position:
            return True, 1
        return False        

class Player(ABC):
    @abstractmethod
    def evaluate(position : Game):
        pass

    @abstractmethod
    def play(position : Game, time : float):
        pass
