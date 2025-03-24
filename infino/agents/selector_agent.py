
import torch
import torch.nn as nn
import torch.optim as optim

class SelectorAgent(nn.Module):
    def __init__(self, input_dim, latent_dim, num_thoughts):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_thoughts = num_thoughts

        # A learned "thought library": num_thoughts x latent_dim
        self.thought_bank = nn.Parameter(torch.randn(num_thoughts, latent_dim))

        # Selector network: chooses which thought vector(s) to activate
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_thoughts),
            nn.Softmax(dim=-1)  # attention weights over thought vectors
        )

        # Decoder maps selected thought to output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        selector_weights = self.selector(x)                    # (batch, num_thoughts)
        selected_thought = selector_weights @ self.thought_bank  # (batch, latent_dim)
        output = self.decoder(selected_thought)                # (batch, 1)
        return output.squeeze(1)

    def train_on_data(self, x, y, epochs=300, lr=0.01):
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(x_tensor)
            loss = loss_fn(y_pred.unsqueeze(1), y_tensor)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        return history

    def predict(self, x):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            y_pred = self.forward(x_tensor)
        return y_pred.numpy()
