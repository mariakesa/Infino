
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CuriousSelectorAgent(nn.Module):
    def __init__(self, input_dim, latent_dim, num_thoughts, temperature=0.5, hard=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_thoughts = num_thoughts
        self.temperature = temperature
        self.hard = hard

        # Thought bank
        self.thought_bank = nn.Parameter(torch.randn(num_thoughts, latent_dim))

        # Selector network
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_thoughts)  # logits
        )

        # Curiosity tracking (moving average of use)
        self.register_buffer("usage_counts", torch.zeros(num_thoughts))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, training=True):
        logits = self.selector(x)

        if training:
            # Encourage curiosity: bias logits toward less-used thoughts
            #curiosity_weight=10.0
            curiosity_bonus = (1.0/(1.0 + self.usage_counts)).to(logits.device)
            #print(curiosity_bonus)
            boosted_logits = logits + curiosity_bonus
        else:
            alpha=0.01
            curiosity_bonus = (1.0/(1.0 + self.usage_counts)).to(logits.device)
            boosted_logits = logits + alpha * curiosity_bonus
        
        sf=nn.Softmax(dim=-1)
        distr = sf(boosted_logits)
        print(distr)
        gumbel_weights = F.gumbel_softmax(boosted_logits, tau=self.temperature, hard=self.hard, dim=-1)
        

        if training:
            # Track which thoughts were used
            chosen_ids = gumbel_weights.argmax(dim=-1)
            for idx in chosen_ids:
                self.usage_counts[idx] += 1

        selected_thought = gumbel_weights @ self.thought_bank
        output = self.decoder(selected_thought)
        return output.squeeze(1)

    def train_on_data(self, x, y, epochs=300, lr=0.01):
        self.usage_counts.zero_()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(x_tensor, training=True)
            loss = loss_fn(y_pred.unsqueeze(1), y_tensor)
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        return history

    def predict(self, x):
        self.eval()
        x_tensor = torch.FloatTensor(x).unsqueeze(1)
        with torch.no_grad():
            y_pred = self.forward(x_tensor, training=False)
        return y_pred.numpy()
