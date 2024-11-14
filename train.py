import argparse
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb

from evaluator import Evaluator
from position import PositionDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--name_in", type=str, default=None)
    parser.add_argument("--name_out", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    
    
    dataset = PositionDataset.load("./data/" + args.dataset)
    if args.name_in is not None:
        model = Evaluator.load_from_checkpoint("./models/" + args.name_in)
    else:
        model = Evaluator(args.dim, args.n_heads, args.n_layers, 1e-3)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, persistent_workers=True, drop_last=True)

    torch.set_float32_matmul_precision('medium')
    logger = WandbLogger(project='checkers', name=args.name_out)
    trainer = Trainer(max_epochs=args.epochs, logger=logger, log_every_n_steps=1, devices=4, enable_progress_bar=False)
    trainer.fit(model, data_loader)
    wandb.finish()
    
    trainer.save_checkpoint("./models/" + args.name_out)