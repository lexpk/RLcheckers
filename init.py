from tqdm import tqdm
from MCTD import MCTD
from argparse import ArgumentParser
from math import exp
from random import randint
import os
from subprocess import Popen, PIPE, STDOUT
import torch

from position import Position, PositionDataset
from checkers import RandomPlayer, Game

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--size', type=int, default=100_000)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--name_out', type=str, default='init.ckpt')
    args = parser.parse_args()
    
    pos = []
    ev = []
    for i in tqdm(range(0, args.size, 1), desc="Generating Random Positions"):
        light_man_cnt = randint(0, 6)
        light_king_cnt = randint(0, 3)
        dark_man_cnt = randint(0, 6)
        dark_king_cnt = randint(0, 3)
        position = Position.random(dark_man_cnt, dark_king_cnt, light_man_cnt, light_king_cnt)
        position.color = 1
        pos.append(position)
        ev.append([[1./(1. + exp(-float(dark_man_cnt + 3*dark_king_cnt - light_man_cnt - 3*light_king_cnt)))]])
    
    datset = PositionDataset(torch.stack([p.nn_input() for p in pos]), torch.tensor(ev))
    os.makedirs("./data/", exist_ok=True)
    datset.save("./data/init.pt")

    with Popen(
        [
            'python', '-u', 'train.py',
            '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
            '--dim', str(args.dim), '--n_layers', str(args.n_layers), '--n_heads', str(args.n_heads),
            '--name_out', args.name_out, '--dataset', 'init.pt'
        ],
        bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
    ) as proc:
        for line in proc.stdout:
            print(line, end='')	