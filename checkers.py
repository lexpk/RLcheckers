from abc import ABC, abstractmethod
from math import exp
import multiprocessing as mp
from random import choice
from time import sleep
from typing import List

from position import Position


class Game():
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
                maxply: maximum number of moves before position is annuled and scored by material count. Used to prevent very long games.
                verbose: If True players might print additional information to the console. 
        '''
        def in_ipynb():
            try:
                cfg = get_ipython().config 
                if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
                    return True
                else:
                    return False
            except NameError:
                return False
        
        if in_ipynb():
            from IPython.display import clear_output

        while self.result == None: 
            if self.position.color == 1:
                self.next(p1, time = movetime, maxply = maxply, verbose = verbose)
            else:
                self.next(p2, time = movetime, maxply = maxply, verbose = verbose)
            if rendering:
                if in_ipynb():
                    clear_output(wait=True)
                print(self.position.ascii())
                if hasattr(p1, 'ev_trace') and p1.ev_trace:
                    print(f"P1 EV: {p1.ev_trace[-1]}")
                if hasattr(p2, 'ev_trace') and p2.ev_trace:
                    print(f"P2 EV: {p2.ev_trace[-1]}")


class Player(ABC):
    '''
    Abstract base class for agents.
    '''
    @abstractmethod
    def move(self, position : Position, time : float, trace : List[Position], verbose : bool):
        pass


class RandomPlayer(Player):
    '''
    Agent that chooses a random move at each position.
    '''
    def __init__(self):
        pass

    def move(self, position : Position, time : float, trace = [], verbose = False):
        sleep(time)
        return choice(position.legal_moves())
