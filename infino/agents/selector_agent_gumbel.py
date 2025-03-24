
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SelectorAgent(nn.Module):
    def __init__(self, input_dim, latent_dim, num_thoughts, temperature=0.5, hard=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_thoughts = num_thoughts
        self.temperature = temperature
        self.hard = hard  # if True, use hard one-hot sampling

        # A learned "thought library": num_thoughts x latent_dim
        self.thought_bank = nn.Parameter(torch.randn(num_thoughts, latent_dim))

        # Selector network: outputs logits for Gumbel-Softmax
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_thoughts)  # logits
        )

        # Decoder maps selected thought to output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        logits = self.selector(x)  # raw logits
        # Apply Gumbel-Softmax sampling
        gumbel_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)
        selected_thought = gumbel_weights @ self.thought_bank  # shape: (batch, latent_dim)
        output = self.decoder(selected_thought)
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
