from typing import List
from xmlrpc.client import Boolean
from numpy import log, sqrt
from numpy.random import default_rng
from math import exp
import torch
from tqdm import tqdm
from checkers import Game, Player
from time import time

class MCTD(torch.nn.Module, Player):

    def __init__(self, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        super(MCTD, self).__init__()
        self.c1 = torch.nn.Conv1d(4, 32, 12, stride=4, device=device)
        self.c2 = torch.nn.Conv1d(32, 16, 1, groups=4 , device=device)
        self.c3 = torch.nn.Conv1d(16, 16, 3, groups=4 , stride=2 , device=device)
        self.c4 = torch.nn.Conv1d(16, 8, 2, device=device)
        self.c5 = torch.nn.Conv1d(8, 1, 1, device=device)
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lossfunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.tree = None
    
    def forward(self, x):
        x = self.c1(x)
        x = self.activation(x)
        x = self.c2(x)
        x = self.activation(x)
        x = self.c3(x)
        x = self.activation(x)
        x = self.c4(x)
        x = self.activation(x)
        x = self.c5(x)
        x = self.sigmoid(x)
        return x

    def bootstrap(self, size = 10000, epochs = 1000):
        positions = []
        material_balances = []
        rng = default_rng()
        for i in tqdm(range(0, int(1.25*size), 1), desc="Generating Random Positions"):
            kings_prob = 0.3*rng.random()
            light_man_cnt = rng.integers(1, 13)
            light_king_cnt = rng.binomial(12 - light_man_cnt, kings_prob)
            dark_man_cnt = rng.integers(1, 13)
            dark_king_cnt = rng.binomial(12 - dark_man_cnt, kings_prob)
            material_balances.append([[ 1./(1. + exp(float(dark_man_cnt + 3*dark_king_cnt - light_man_cnt - 3*light_king_cnt)/exp(1)))]])
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
            positions.append([
                    light_man_pos,
                    light_king_pos,
                    dark_man_pos,
                    dark_king_pos,
                ])
        train_inputs = torch.tensor(positions[:size], dtype=torch.float, device=self.device)
        train_targets = torch.tensor(material_balances[:size], dtype=torch.float, device=self.device)
        test_inputs = torch.tensor(positions[size:], dtype=torch.float, device=self.device)
        test_targets = torch.tensor(material_balances[size:], dtype=torch.float, device=self.device)

        for epoch in tqdm(range(epochs), desc="Training Model"):
            self.zero_grad()
            prediction = self(train_inputs)
            loss = self.lossfunction(prediction, train_targets)
            loss.backward()
            self.optimizer.step()
        
        with torch.no_grad():
            prediction = self(train_inputs)
            loss = self.lossfunction(prediction, train_targets)
            print(f"Performance on trainset: {loss}")
            prediction = self(test_inputs)
            loss = self.lossfunction(prediction, test_targets)
            print(f"Performance on testset: {loss}")

    def to_file(self, name):
        torch.save(self.state_dict(), f".\\models\\{name}")

    def from_file(name):
        m = MCTD()
        m.load_state_dict(torch.load(f".\\models\\{name}"))
        return m

    def move(self, game : Game, timelimit, learn = False):
        t_0 = time()
        if self.tree == None or self.tree.position != game.position:
            self.tree = VariationTree(None, None, game.position, game.color)
        self.tree.expand(4, game.trace)
        self.tree.expansions = 1
        nodes = self.tree.collect_ev_nodes()
        with torch.no_grad():
            VariationTree.ev_nodes(self, nodes)
        self.tree.update_ev()
        ds = []
        e = 0
        while(time() - t_0 < timelimit):
            e += 1
            node = self.tree
            d = 4
            while(node.expansions):
                if node.terminal:
                    break
                if all(c.terminal for c in node.children):
                    node.terminal = True
                    break
                d += 1 
                node = max(
                    [c for c in node.children if not c.terminal],
                    key = lambda c : (1 - c.ev) + sqrt(2. * log(node.expansions)/(1+c.expansions))
                )
                node.parent.unev_children = True
                node.parent.expansions += 1
            ds.append(d)
            node.expand(4, game.trace)
            node.expansions = 1
            nodes = self.tree.collect_ev_nodes()
            with torch.no_grad():
                VariationTree.ev_nodes(self, nodes)
            self.tree.update_ev()
        print(f"{self.tree.ev} at maxdepth {max(ds) if ds else 4} with {e} expansions")
        child = min(self.tree.children, key=lambda c : c.ev)
        self.tree = child.reroot(child)
        return self.tree.position

class VariationTree():

    def __init__(self, root, parent, position : List[int], color : int):
        self.root : VariationTree = root
        self.parent : VariationTree = parent
        self.position :  List[int] = position
        self.color : int = color
        self.expansions : int = 0
        self.leaf : Boolean = True
        self.ev : float = -1.
        self.children : List[VariationTree] = []
        self.unev_children : Boolean = False
        self.terminal = False
        if self.root == None:
            self.root = self
            self.parent = None

    def expand(self, d : int, trace = []):
        if (self.position, self.color) in trace:
            self.ev = 0.5
        else:
            if d > 0:
                if self.leaf:
                    for pos in Game.legal_moves(self.position, self.color):
                        self.children.append(VariationTree(self.root, self, pos, -self.color))
                        self.leaf = False
                for child in self.children:
                    child.expand(d - 1, trace + [(self.position, self.color)])
                    self.unev_children = True
                if any(c.ev == 0 for c in self.children) or all(c.terminal for c in self.children):
                    self.terminal = True
                    if self.children:    
                        self.ev = 1 - min(c.ev for c in self.children)
                    else:
                        self.ev = 0
            else:
                position = [-piece for piece in self.position[::-1]] if self.color == -1 else self.position
                if any(Game._legal_dark_single_captures_no_promotion(piece, position) for piece in range(32)):
                    self.expand(1)

    def collect_ev_nodes(self):
        nodes = []
        if self.ev == -1:
            nodes.append(self)
        if self.unev_children:
            for child in self.children:
                nodes += child.collect_ev_nodes()
        return nodes

    def ev_nodes(nn, nodes):
        if nodes:
            inputs = torch.tensor(
                [
                    [
                        [position[i] == 1 for i in range(32)],
                        [position[i] == 2 for i in range(32)],
                        [position[i] == -1 for i in range(32)],
                        [position[i] == -2 for i in range(32)],
                    ] for position in map(lambda x : x.position if x.color == 1 else Game.flip(x.position), nodes)
                ],
                dtype=torch.float,
                device = nn.device
            )
            evs = nn(inputs)
            for i, node in enumerate(nodes):
                node.ev = (float)(evs[i])
    
    def update_ev(self):
        if self.unev_children:
            for child in self.children:
                child.update_ev()
            self.ev = 1 - min(child.ev for child in self.children)
    
    def reroot(self, root):
        self.root = root
        self.ev = -1
        for child in self.children:
            child.reroot(root)
        return root

        
