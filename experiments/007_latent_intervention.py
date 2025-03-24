
import torch
import numpy as np
import matplotlib.pyplot as plt

from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.mimicker_patchable import MimickerAgent
from infino.agents.sparse_autoencoder import SparseAutoencoder

# 1. Simulate world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=200)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# 2. Train Mimicker
agent = MimickerAgent()
agent.train_on_data(x, y, epochs=300, lr=0.01)

# 3. Get hidden activations
hidden = agent.get_hidden_activations(x)

# 4. Train Sparse Autoencoder
autoencoder = SparseAutoencoder(input_dim=hidden.shape[1], latent_dim=4, sparsity_lambda=1e-3)
autoencoder.train_on_data(hidden, epochs=300, lr=0.01)

# 5. Encode to latent
h_tensor = torch.FloatTensor(hidden)
_, z = autoencoder(h_tensor)

# 6. Modify one latent dimension (zero it out)
z_mod = z.clone()
z_mod[:, 0] = 0  # zero out latent dim 0

# 7. Decode modified latents back to activations
patched_activations = autoencoder.decoder(z_mod).detach()

# 8. Predict using patched activations
y_pred_mod = agent.predict(x, patched_activations=patched_activations)
y_pred_orig = agent.predict(x)

# 9. Plot results
plt.figure(figsize=(10, 5))
plt.plot(y, label='True $x_{t+1}$', alpha=0.7)
plt.plot(y_pred_orig, label='Original Prediction', linestyle='--')
plt.plot(y_pred_mod, label='Modified Latent Prediction (dim 0 = 0)', linestyle='--')
plt.title("Effect of Latent Intervention on Prediction")
plt.xlabel("Time Step")
plt.ylabel("Next Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot latent dimensions over time
plt.figure(figsize=(10, 6))
for i in range(patched_activations.shape[1]):
    plt.plot(patched_activations[:, i], label=f"Latent dim {i+1}")
plt.title("Sparse Autoencoder Latent Dimensions")
plt.xlabel("Time Step")
plt.ylabel("Activation")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
