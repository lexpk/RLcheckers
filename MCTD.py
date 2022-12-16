from math import exp
from numpy import log, sqrt
from numpy.random import default_rng
from time import time
import torch
from tqdm import tqdm
from typing import List

from position import Position
from checkers import Game, Player

class MCTD(torch.nn.Module, Player):
    '''
    The main class tracking the logic of the reinforcement learning agent.
    '''
    def __init__(self, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        '''
        Initializes a new RL Agent.

                Parameters:
                    device: The device the training is to be performed on.
        '''
        self.device = device
        super(MCTD, self).__init__()
        self.c1 = torch.nn.Conv1d(4, 8, 12, stride=4, device=torch.device('cpu'))
        self.c2 = torch.nn.Conv1d(8, 1, 1, device=torch.device('cpu'))
        self.c3 = torch.nn.Linear(6, 1, device=torch.device('cpu'))
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lossfunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.tree = None
        self.trace = []
        self.ev_trace = []
        self.eval()
    
    def forward(self, x):
        x = self.c1(x)
        x = self.activation(x)
        x = self.c2(x)
        x = self.activation(x)
        x = self.c3(x)
        x = self.sigmoid(x)
        return x

    def bootstrap(self, size : int = 25000, epochs : int = 2500):
        '''
        Initializes the weights by estimating the material balance of random positions.

            Parameters:
                size: number of random positions to train on
                epochs: number of epochs to train on (empirically size/10 works well)        
        '''
        rng = default_rng()
        pos = []
        ev = []
        for i in tqdm(range(0, size, 1), desc="Generating Random Positions"):
            light_man_cnt = rng.integers(1, 6)
            light_king_cnt = rng.integers(1, 6)
            dark_man_cnt = rng.integers(1, 6)
            dark_king_cnt = rng.integers(1, 6)
            position = Position.random(dark_man_cnt, dark_king_cnt, light_man_cnt, light_king_cnt)
            position.color = 1
            pos.append(position)
            ev.append([[1./(1. + exp(-float(dark_man_cnt + 3*dark_king_cnt - light_man_cnt - 3*light_king_cnt)))]])
        self.learn(pos, ev, epochs=epochs)

    def learn(self, pos : List[Position], ev : List[float], epochs : int = 1000):
        '''
        updates weights according to provided data.

            Parameters:
                pos: List of positions to learn from
                ev: List of correct evaluations for positions in pos
        '''
        self.to(device=self.device)
        self.train()

        inputs = torch.tensor(
            [p.nn_input() for p in pos],
            dtype=torch.float,
            device=self.device
        )
        ev = torch.tensor(
            ev,
            dtype=torch.float,
            device=self.device
        )
        for epoch in tqdm(range(epochs), desc="Learning", leave = False):
            self.zero_grad()
            prediction = self(inputs)
            loss = self.lossfunction(prediction, ev)
            loss.backward()
            self.optimizer.step()

        self.cpu()
        self.eval()
       

    def to_file(self, name):
        '''
        Stores model to file.
        '''
        torch.save(self.state_dict(), f".\\models\\{name}")

    def from_file(name):
        '''
        Loads model from file
        '''
        m = MCTD()
        m.load_state_dict(torch.load(f".\\models\\{name}"))
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
        t_0 = time()
        if self.tree == None:
            self.tree = VariationTree(pos)
        else:
            for child in self.tree.children:
                if pos == child.position:
                    self.tree = child
                    break
            else:
                self.tree = VariationTree(pos)
        while (time() - t_0 < timelimit) and not self.tree.dead:
            self.tree.expand(self, trace)
        if verbose:
            print(f"{float(self.tree.ev)} with {self.tree.expansions} expansions")      
        self.trace.append(self.tree.position)
        self.ev_trace.append(self.tree.ev)
        child = max(self.tree.children, key=lambda c : 1 - c.ev)
        self.tree = child
        return self.tree.position

    def learn_by_selfplay(self, positions, movetime, lmb = 0.9, epochs=100, rendering = False):
        '''
        Adjusts weights by playing games against itself and performing TD(lambda) learning

            Parameters:
                positions: starting positions for selfplay.
                movetime: maximum thinking time per move
                lmb: lambda for TD(lambda) (0.9 worked ok in some experiments)
                epochs: number of epochs for learning weights (again n/10 worked well where n ~ positions to learn from)
                rendering: If true displays the game in gui during selfplay
        '''
        pos = []
        ev = []
        result = 0
        for position in tqdm(positions, desc="Playing"):
            game = Game(position)
            game.simulate(self, self, movetime=movetime, rendering=rendering, maxply = 200, verbose = False)
            e = self.ev_trace[-1]
            for i in range(len(self.trace) - 1):
                e =  lmb * (1 - e) + (1 - lmb) * self.ev_trace[-1-i]
                self.ev_trace[-1-i] = e
            pos += self.trace
            ev += self.ev_trace
            self.trace = []
            self.ev_trace = []
        self.learn(pos, [[[x]] for x in ev], epochs=epochs)

class VariationTree():
    '''
    Internal class for keeping track of the consdiered Variations
    '''
    def __init__(self, position : Position):
        '''
        Initializes new Variation tree with a single node.

            Parameters:
                position: position stored in the single node
        '''
        self.position :  Position = position
        self.expansions : int = 0
        self.dead = False
        self.ev = None
        self.children : List[VariationTree] = []

    def expand(self, player, trace = []):
        '''
        Expands the search tree according to exploration-exploitation tradeoff as given in the file.

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
                child.expand(player, trace + [self.position])
                self.ev = max((1 - c.ev for c in self.children), default = 0)
                self.expansions += 1
                self.dead = all(c.dead for c in self.children) or any(c.ev == 0 for c in self.children)
            else:
                self.dead = True
        else:
            if self.position in trace:
                self.ev = torch.tensor([[0.5]])
                self.dead = True
            else:
                for pos in self.position.legal_moves():
                    if pos not in [c.position for c in self.children]:
                        child = VariationTree(pos)
                        child.eval(player)
                        self.children.append(child)     
                self.ev = max((1 - c.ev for c in self.children), default=0)
                self.expansions += 1           
                self.dead = all(c.dead for c in self.children) or any(c.ev == 0 for c in self.children)
    
    def eval(self, player):
        '''
        evaluates a node w.r.t. a evaluation function.

            Parameters:
                player: evaluation function for positions
        '''
        if len(self.position.legal_moves()) == 0:
            self.ev = torch.tensor([[0]])
            self.dead = True
            return
        if self.position.has_captures():
            self.expand(player, [])
            self.ev = max((1 - c.ev for c in self.children), default=0)
            self.dead = all(c.dead for c in self.children) or any(c.ev == 0 for c in self.children)
            return
        with torch.no_grad():
            self.ev = player.forward(
                torch.tensor(
                    self.position.nn_input(),
                    dtype=torch.float,
                    device=torch.device('cpu')
                )
            )

        
