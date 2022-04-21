from typing import List
from numpy import argmax
from numpy.random import default_rng
from math import exp
import torch
from tqdm import tqdm
from TensorRepresentation import TensorRepresentation
from checkers import Player
from time import time

class MCTD(torch.nn.Module, Player):

    def __init__(self, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        super(MCTD, self).__init__()
        self.c1 = torch.nn.Conv1d(6, 32, 12, stride=4, device=device)
        self.c2 = torch.nn.Conv1d(32, 32, 1, groups=4 , device=device)
        self.c3 = torch.nn.Conv1d(32, 16, 1, groups=4 , device=device)
        self.c4 = torch.nn.Conv1d(16, 16, 3, groups=4 , stride=2 , device=device)
        self.c5 = torch.nn.Conv1d(16, 8, 2, device=device)
        self.c6 = torch.nn.Conv1d(8, 1, 1, device=device)
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lossfunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
    
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
        x = self.activation(x)
        x = self.c6(x)
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
                    [x or y for x, y in zip(light_man_pos, light_king_pos)],
                    dark_man_pos,
                    dark_king_pos,
                    [x or y for x, y in zip(dark_man_pos, dark_king_pos)]
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

    def move(self, game, time):
        pass


class GameTree:

    def __init__(self, root, prev, loc):
        self.root = root
        self.prev = prev
        self.loc = loc
        self.expansions = -1
        self.children : List[GameTree] = []
        self.pos = None
        self.nnev = None
        self.ev = -1.

    def from_position(position):
        i = 0
        t = GameTree(None, None, i)
        t.pos = position.expand(1, 6, 32)
        i += 1
        nextpos = TensorRepresentation.next_positions(t.pos)
        last = i
        t.pos = torch.cat((t.pos, nextpos[0]))
        for _ in range(len(nextpos[0])):
            t.children.append(GameTree(t, t, i))
            t.expansions += 1
            i += 1
        nextpos = TensorRepresentation.next_positions(t.pos[last:i])
        last = i
        for p in nextpos:
            t.pos = torch.cat((t.pos, p))
        leaf_i = 0
        for t_1 in t.children:
            t_1.expansions = 0
            for _ in nextpos[leaf_i]:
                t_1.children.append(GameTree(t, t_1, i))
                i += 1
            leaf_i += 1
        nextpos = TensorRepresentation.next_positions(t.pos[last:i])
        last = i
        for p in nextpos:
            t.pos = torch.cat((t.pos, p))
        leaf_i = 0
        for t_1 in t.children:
            for t_2 in t_1.children:
                for _ in nextpos[leaf_i]:
                    t_2.children.append(GameTree(t, t_2, i))
                    i += 1
                leaf_i += 1
        nextpos = TensorRepresentation.next_positions(t.pos[last:i])
        last = i
        for p in nextpos:
            t.pos = torch.cat((t.pos, p))
        leaf_i = 0
        for t_1 in t.children:
            for t_2 in t_1.children:
                for t_3 in t_2.children:
                    for _ in nextpos[leaf_i]:
                        t_3.children.append(GameTree(t, t_3, i))
                        i += 1
                    leaf_i += 1
        nextpos = TensorRepresentation.next_positions(t.pos[last:i])
        last = i
        for p in nextpos:
            t.pos = torch.cat((t.pos, p))
        leaf_i = 0
        for t_1 in t.children:
            for t_2 in t_1.children:
                for t_3 in t_2.children:
                    for t_4 in t_3.children:
                        for _ in nextpos[leaf_i]:
                            t_4.children.append(GameTree(t, t_4, i))
                            i += 1
                        leaf_i += 1
        return t

    def expand(self):
        fr = self.children[0].children[0].children[0].children[0].loc
        to = self.children[-1].children[-1].children[-1].children[-1].loc + 1
        nextpos = TensorRepresentation.next_positions(self.root.pos[fr:to])
        i = len(self.root.pos)
        for p in nextpos:
            self.root.pos = torch.cat((self.root.pos, p))
        leaf_i = 0
        for t_1 in self.children:
            for t_2 in t_1.children:
                for t_3 in t_2.children:
                    for _ in nextpos[leaf_i]:
                        t_3.children.append(GameTree(self.root, t_3, i))
                        i += 1
                    leaf_i += 1

    def make_nnev(self, player : MCTD):
        assert self.pos != None, "Game Tree not expanded" 
        self.nnev = player(self.pos)
    
    def get_ev(self):
        if self.children == []:
            self.ev = self.root.nnev[self.loc]
        else:
            self.ev = 1 - max([child.get_ev() for child in self.children])
        return self.ev
