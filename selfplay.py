from MCTD import MCTD
from argparse import ArgumentParser
import time
import torch

from position import Position
from checkers import RandomPlayer, Game


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--name', type=str, default='model.ckpt')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--lmb', type=float, default=0.9)
    args = parser.parse_args()
    
    p = MCTD.from_file(args.name)
    r = RandomPlayer()
    game = Game(r.move(r.move(Position(), 0), 0))
    game.simulate(p, p, movetime=args.movetime, rendering=False, maxply=120, verbose=False)
    e = p.ev_trace[-1]
    for i in range(len(p.trace) - 1):
        e = args.lmb * (1 - e) + (1 - args.lmb) * p.ev_trace[-1-i]
        p.ev_trace[-1-i] = e
    torch.save((
        torch.stack(
            [pos.nn_input() for pos in p.trace],
        ), torch.tensor(p.ev_trace)
    ), f"./games/stage_{args.stage}/{time.time()}.pt")
