from MCTD import MCTD
from argparse import ArgumentParser
import os
from subprocess import Popen, PIPE, STDOUT
import torch

from position import Position, PositionDataset
from checkers import RandomPlayer, Game

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--name', type=str, default='model.ckpt')
    parser.add_argument('--name_out', type=str, default='model.ckpt')
    parser.add_argument('--period', type=int, default=1)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--lmb', type=float, default=0.9)
    args = parser.parse_args()
    
    os.makedirs(f".\\games\\data_{args.period}", exist_ok=True)
    procs = [
        Popen(
            [
                'python', '-u', 'selfplay',
                '--movetime', str(args.movetime), '--name', args.name,
                '--period', str(args.period), '--lmb', str(args.lmb)
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ) for _ in range(args.size)
    ]
    for proc in procs:
        proc.wait()
    
    player = MCTD.from_file(args.name)
    data_list = [
        torch.load(f".\\games\\data_{args.period}\\{file}", weights_only=True)
        for file in os.listdir(f".\\data_{args.period}")
    ]
    dataset = PositionDataset()
    dataset.positions = torch.cat([data[0] for data in data_list])
    dataset.evaluations = torch.cat([data[1] for data in data_list])
    dataset.save(f".\\data_{args.period}\\dataset.pt")