
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MimickerAgent(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, patched_activations=None):
        h = self.fc1(x)
        a = self.relu(h)
        if patched_activations is not None:
            a = patched_activations
        out = self.fc2(a)
        return out

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

    def predict(self, x, patched_activations=None):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            out = self.forward(x_tensor, patched_activations=patched_activations)
        return out.squeeze(1).numpy()

    def get_hidden_activations(self, x):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            h = self.fc1(x_tensor)
            a = self.relu(h)
        return a.numpy()
