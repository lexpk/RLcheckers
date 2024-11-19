This is an implementation of a simple checkers agent based on transformers, MCTS and TD(λ). For more information please read the [report](report/report.pdf).

### Requirements

Running the training and evaluation requires pytorch, lightning and wandb. Watching the agents play in [experiments.ipynb](experiments.ipynb)  additionally requires ipykernel.

### Installation and use

To train the agents run
```
./train.sh
```
Afterwards, you can run the tournament described in the report via
```
./tournament.sh
```
or observe agents play in [experiments.ipynb](experiments.ipynb).
