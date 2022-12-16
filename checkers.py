from abc import ABC, abstractmethod
from math import exp
import multiprocessing as mp
from numpy.random import default_rng
import PySimpleGUI as sg
from time import sleep
from typing import List

from position import Position

def visualize(position):
    '''
    Opens new window with visual representation of position.
    '''
    layout = [
        [
            sg.Image(
                Game.IMAGE[3 + position.squares[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0] , subsample=5, key=(i, j)
            ) for j in range(8)
        ]
        for i in range(7, -1, -1)
    ]   
    sg.Window('Checkers', layout, size=(864, 832)).read(close=True)

def _windowloop(window, receiver):
    '''
    Internal function used by the subprocess that manages the window showing an active game.
    '''
    while True:
        event, _ = window.read(timeout=10)
        if event != '__TIMEOUT__':
            return
        if receiver.poll():
            position = receiver.recv()
            for i in range(7, -1, -1):
                for j in range(8):
                    window[(i, j)].update(
                        Game.IMAGE[3 + position.squares[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0],
                        subsample=5
                    )

class Game():
    '''
    Class for keeping track of an ongoing game.
    '''
    IMAGE = {}
    IMAGE[0] = ".\\images\\lightsquare_empty.png"
    IMAGE[1] = ".\\images\\darksquare_light_king.png"
    IMAGE[2] = ".\\images\\darksquare_light_man.png"
    IMAGE[3] = ".\\images\\darksquare_empty.png"
    IMAGE[4] = ".\\images\\darksquare_dark_man.png"
    IMAGE[5] = ".\\images\\darksquare_dark_king.png"


    def __init__(self, position = Position(), ply = 0):
        '''
        Initializes new game.

            Parameters: 
                position: starting position (default: starting position)
                ply: number of (half)moves already played
        '''
        self.position = position
        self.ply = ply
        self.result = None
        self.trace = []
        self.rendering = False
 

    def render(self):
        '''
        Opens a new window in which the game is displayed and updated live.
        '''
        self.sender, receiver = mp.Pipe()
        layout = [
            [
                sg.Image(
                    Game.IMAGE[3 + self.position.squares[4*i + j//2]] if (i + j)%2 == 0 else Game.IMAGE[0] , subsample=5, key=(i, j)
                ) for j in range(8)
            ]
            for i in range(7, -1, -1)
        ]   
        window = sg.Window('Checkers', layout, size=(864, 832))
        self.windowprocess =  mp.Process(target = _windowloop, args=[window, receiver])
        self.windowprocess.start()
        self.rendering = True          

    def next(self, player, time : float, maxply = 10000, verbose = False):
        '''
        Prompts player for move and progesses the game accordingly

            Parameters:
                player: The player to prompt
                time: timelimit for the player (is not enforced)
                maxply: maximum number of moves before position is annuled and scored by material count. Used to prevent very long games.
                verbose: If True player might print additional information to the console.
        '''
        next = player.move(self.position, time, trace = self.trace, verbose = verbose)
        assert next in self.position.legal_moves()
        self.ply += 1
        
        self.trace.append(self.position)
        self.position = next
        if not next.legal_moves():
            self.result = -self.position.color
        if self.position in self.trace:
            self.result = 0
        if self.ply > maxply:
            self.result = 1./(1. + exp(-float(sum([
                len([x for x in self.position.squares if x == 1]),
                3*len([x for x in self.position.squares if x == 2]),
                -len([x for x in self.position.squares if x == -1]),
                -3*len([x for x in self.position.squares if x == -2])
            ]))))

    def simulate(self, p1, p2 , movetime = 1, rendering = True, maxply = 10000, verbose = False):
        '''
        Simulates game between two players.

            Parameters:
                p1: player of the dark pieces
                p2: player of the light pieces
                movetime: maximum thinking time per move (not enforced)
                rendering: If True opens new window to display game, updating live as moves are occurring
                maxply: maximum number of moves before position is annuled and scored by material count. Used to prevent very long games.
                verbose: If True players might print additional information to the console. 
        '''
        if rendering:
            self.render()
        while self.result == None:          
            if self.position.color == 1:
                self.next(p1, time = movetime, maxply = maxply, verbose = verbose)
            else :
                self.next(p2, time = movetime, maxply = maxply, verbose = verbose)
            if rendering:    
                self.sender.send(self.position)

class Player(ABC):
    '''
    blueprint for player class
    '''
    @abstractmethod
    def move(self, position : Position, time : float, trace : List[Position], verbose : bool):
        pass

class RandomPlayer(Player):
    '''
    Agent that chooses a random move at each position.
    '''
    def __init__(self):
        self.rng = default_rng()

    def move(self, position : Position, time : float, trace = [], verbose = False):
        sleep(time)
        return self.rng.choice(position.legal_moves())
