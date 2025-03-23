
import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            #nn.Sigmoid()
        )
        self.sparsity_lambda = sparsity_lambda

        # Custom initialization (Kaiming for ReLU layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def train_on_data(self, x, epochs=200, lr=0.01):
        x_tensor = torch.FloatTensor(x)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            x_recon, z = self.forward(x_tensor)

            recon_loss = loss_fn(x_recon, x_tensor)
            sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
            total_loss = recon_loss + sparsity_loss

            total_loss.backward()
            optimizer.step()
            history.append(total_loss.item())

        return history
