
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MimickerAgent(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

    def train_on_data(self, x, y, epochs=200, lr=0.01):
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        return history

    def predict(self, x):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            y_pred = self.forward(x_tensor).squeeze(1).numpy()
        return y_pred

    def get_hidden_activations(self, x):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            hidden = self.model[0](x_tensor)  # Linear
            activated = self.model[1](hidden) # ReLU
        return activated.numpy()