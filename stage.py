from MCTD import MCTD
from argparse import ArgumentParser
import os
from subprocess import Popen, PIPE, STDOUT
import torch
from tqdm import tqdm

from position import PositionDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--name_in', type=str, default='model.ckpt')
    parser.add_argument('--name_out', type=str, default='model.ckpt')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--lmb', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    os.makedirs(f"./games/stage{args.stage}", exist_ok=True)
    
    for _ in tqdm(range(args.size), desc=f"Stage {args.stage}"):
        Popen(
            [
                'python', '-u', 'selfplay.py',
                '--movetime', str(args.movetime), '--name', args.name_in,
                '--stage', str(args.stage), '--lmb', str(args.lmb)
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ).wait()

    data_list = [
        torch.load(f"./games/stage{args.stage}/{file}", weights_only=True)
        for file in os.listdir(f"./games/stage{args.stage}")
    ]
    dataset = PositionDataset(
        torch.cat([data[0] for data in data_list]),
        torch.cat([data[1] for data in data_list])
    )
    dataset.save(f"./stage{args.stage}.pt")

    with Popen(
        [
            'python', '-u', 'train.py',
            '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
            '--name_in', args.name_in, '--name_out', args.name_out, '--dataset', f"stage{args.stage}.pt"
        ],
        bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
    ) as proc:
        for line in proc.stdout:
            print(line, end='')
