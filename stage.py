from MCTD import MCTD
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import os
from subprocess import Popen, PIPE, STDOUT
import torch

from position import PositionDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--name', type=str, default='model.ckpt')
    parser.add_argument('--name_out', type=str, default='model.ckpt')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--lmb', type=float, default=0.9)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(f"./games/stage_{args.stage}", exist_ok=True)
    
    def generate_game(a):
        Popen(
            [
                'python', '-u', 'selfplay.py',
                '--movetime', str(a.movetime), '--name', a.name,
                '--stage', str(a.stage), '--lmb', str(a.lmb)
            ],
            bufsize=1, universal_newlines=True, stdout=PIPE, stderr=STDOUT
        ).wait()

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        executor.map(generate_game, [args for _ in range(args.size)])

    data_list = [
        torch.load(f"./games/stage_{args.stage}/{file}", weights_only=True)
        for file in os.listdir(f"./games/stage_{args.stage}")
    ]
    dataset = PositionDataset(
        torch.cat([data[0] for data in data_list]),
        torch.cat([data[1] for data in data_list])
    )
    dataset.save(f"./stage_{args.stage}.pt")