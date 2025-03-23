
import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.mimicker import MimickerAgent
from infino.agents.sparse_autoencoder_ import SparseAutoencoder
import torch

# Step 1: Simulate world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=300)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# Step 2: Train Mimicker
mimicker = MimickerAgent()
mimicker.train_on_data(x, y, epochs=300, lr=0.01)

# Step 3: Extract hidden activations
hidden = mimicker.get_hidden_activations(x)

#print(hidden.shape)
print(hidden)

plt.plot(hidden)
plt.show()

# Step 4: Train Sparse Autoencoder
autoencoder = SparseAutoencoder(input_dim=hidden.shape[1], latent_dim=4, sparsity_lambda=1e-3)
losses = autoencoder.train_on_data(hidden, epochs=300, lr=0.01)

# Step 5: Visualize latent codes
hidden_tensor = torch.FloatTensor(hidden)
_, latent_codes = autoencoder.forward(hidden_tensor)
latent_codes = latent_codes.detach().numpy()

# Plot latent dimensions over time
plt.figure(figsize=(10, 6))
for i in range(latent_codes.shape[1]):
    plt.plot(latent_codes[:, i], label=f"Latent dim {i+1}")
plt.title("Sparse Autoencoder Latent Dimensions")
plt.xlabel("Time Step")
plt.ylabel("Activation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
