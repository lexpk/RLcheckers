import argparse
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import wandb

from MCTD import MCTD
from checkers import RandomPlayer, Game
from position import Position
from cake import Cake


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player1", type=str, default='random')
    parser.add_argument("--player2", type=str, default='random')
    parser.add_argument('--random_moves', type=int, default=4)
    parser.add_argument('--movetime', type=float, default=1)
    parser.add_argument('--maxply', type=int, default=100)
    parser.add_argument('--positions', type=int, default=32)
    args = parser.parse_args()
    
    match args.player1:
        case 'random':
            player1 = RandomPlayer()
        case 'cake':
            player1 = Cake()
        case _:
            player1 = MCTD.from_file(args.player1)

    match args.player2:
        case 'random':
            player2 = RandomPlayer()
        case 'cake':
            player2 = Cake()
        case _:
            player2 = MCTD.from_file(args.player2)
    
    r = RandomPlayer()
    p1wins, p2wins, draws = 0, 0, 0
    for _ in (pbar := tqdm(range(args.positions), desc=f"{args.player1} wins: 0, {args.player2} wins: 0, Draws: 0")):
        starting_position = Position()
        for _ in range(args.random_moves):
            starting_position = r.move(starting_position, 0)
        game = Game(position=starting_position)
        game.simulate(player1, player2, rendering=False, verbose=False)
        if game.result == 1:
            p1wins += 1
        elif game.result == -1:
            p2wins += 1
        else:
            draws += 1
        pbar.set_description(f"{args.player1} wins: {p1wins}, {args.player2} wins: {p2wins}, Draws: {draws}")
        game = Game(position=starting_position)
        game.simulate(player2, player1, rendering=False, verbose=False)
        if game.result == 1:
            p2wins += 1
        elif game.result == -1:
            p1wins += 1
        else:
            draws += 1
        pbar.set_description(f"{args.player1} wins: {p1wins}, {args.player2} wins: {p2wins}, Draws: {draws}")