from math import exp, log, sqrt
from subprocess import Popen, PIPE, STDOUT
from time import time
import torch
from tqdm import tqdm
from typing import List

from position import Position, PositionDataset
from checkers import Game, Player
from evaluator import Evaluator


class MCTD(torch.nn.Module, Player):
    '''
    The main class tracking the logic of the reinforcement learning agent.
    '''
    def __init__(self):
        '''
        Initializes a new RL Agent.

                Parameters:
                    device: The device the training is to be performed on.
        '''
        super(MCTD, self).__init__()
        self.evaluator = None
        self.tree = None
        self.trace = []
        self.ev_trace = []
        self.eval()
    
    def forward(self, x):
        '''
        Forward pass of the model.

            Parameters:
                x: input tensor

            Returns:
                logit of predicted win probability
        '''
        return self.evaluator(x)

    def to_file(self, name):
        '''
        Stores model to file.
        '''
        self.evaluator.save(f"./models/{name}")

    def from_file(name):
        '''
        Loads model from file
        '''
        m = MCTD()
        m.evaluator = Evaluator.load_from_checkpoint(f"./models/{name}").eval()
        return m

    def move(self, pos : Position, timelimit, trace = [], verbose = True):
        '''
        Promt the agent for a single move.

            Parameters:
                pos: position of interest
                timelimit: maximum time for the agent to think
                trace: previous positions for draw evaluation
                verbose: If true agent prints number of expanded nodes during tree search and internal evalulation

            Returns:
                Position after move chosen by the agent is performed
        '''
        self.evaluator.eval()
        t_0 = time()
        self.tree = VariationTree(pos)
        while (time() - t_0 < timelimit) and not self.tree.dead:
            self.tree.expand(self, trace, timelimit, t_0)
        if verbose:
            print(f"{float(self.tree.ev)} with {self.tree.expansions} expansions")      
        self.trace.append(self.tree.position)
        self.ev_trace.append(self.tree.ev)
        child = max(self.tree.children, key=lambda c : 1 - c.ev)
        return child.position


class VariationTree:
    '''
    Internal class for keeping track of the considered Variations
    '''
    def __init__(self, position : Position):
        '''
        Initializes new Variation tree with a single node.

            Parameters:
                position: position stored in the single node
        '''
        self.position: Position = position
        self.expansions: int = 0
        self.dead = False
        self.ev = None
        self.children: List[VariationTree] = []

    def __len__(self):
        if not self.children:
            return 1
        return 1 + sum(len(c) for c in self.children)

    def expand(self, player, trace = [], timelimit=None, t_0=None):
        '''
        Expands the search tree according to exploration-exploitation tradeoff as given in the report.

            Parameters:
                player: evaluation function for positions
                trace: previous positions for draw evaluation
        '''
        if self.expansions > 0:
            if [c for c in self.children if not c.dead]:
                child = max(
                    [c for c in self.children if not c.dead],
                    key = lambda c : (1 - c.ev) + sqrt(2. * log(self.expansions)/(1e-3+c.expansions))
                )
                if timelimit and time() - t_0 > timelimit:
                    return
                child.expand(player, trace + [self.position], timelimit=timelimit, t_0=t_0)
                self.ev = max((1 - c.ev for c in self.children), default = 0)
                self.expansions += 1
                self.dead = all(c.dead for c in self.children) or any(c.ev == 0 for c in self.children)
            else:
                self.dead = True
        else:
            if self.position in trace:
                self.ev = 0.5
                self.dead = True
            else:
                for pos in self.position.legal_moves():
                    if timelimit and time() - t_0 > timelimit:
                        break
                    if pos not in set(c.position for c in self.children):
                        child = VariationTree(pos)
                        child.eval(player)
                        self.children.append(child)
                self.ev = max((1 - c.ev for c in self.children), default=0)
                self.expansions += 1
                self.dead = (self.ev == 0) or (self.ev == 1)

    def eval(self, player):
        '''
        evaluates a node w.r.t. a evaluation function.

            Parameters:
                player: evaluation function for positions
        '''
        if len(self.position.legal_moves()) == 0:
            self.ev = 0
            self.dead = True
            return
        if self.position.has_captures() and \
            not 2 in self.position.squares and \
            not -2 in self.position.squares:
            self.expand(player, player.trace)
            self.ev = max((1 - c.ev for c in self.children), default=0)
            self.dead = all(c.dead for c in self.children) or any(c.ev == 0 for c in self.children)
            return
        else:
            with torch.no_grad():
                self.ev = torch.sigmoid(player.forward(
                    self.position.nn_input()
                )).item()
