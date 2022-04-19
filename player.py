from numpy import dtype
import torch
from tqdm import tqdm

class Player(torch.nn.Module):

    def __init__(self, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Player, self).__init__()
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

    def bootstrap(self, data : torch.tensor, epochs : int = 1):
        for epoch in tqdm(range(epochs)):
            for position, evaluation in data:
                self.zero_grad()
                prediction = self(position)
                loss = self.lossfunction(prediction, evaluation)
                loss.backward()
                self.optimizer.step()
