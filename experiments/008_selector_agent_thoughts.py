
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.selector_agent_gumbel import SelectorAgent

# 1. Simulate world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=300)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# 2. Initialize selector agent
agent = SelectorAgent(
    input_dim=1,
    latent_dim=8,
    num_thoughts=12,
    temperature=0.2,
    hard=True
)

# 3. Train agent on next-state prediction
losses = agent.train_on_data(x, y, epochs=300, lr=0.01)

# 4. Predict and visualize
y_pred = agent.predict(x)

plt.figure(figsize=(10, 5))
plt.plot(y, label="True $x_{t+1}$", alpha=0.8)
plt.plot(y_pred, label="Prediction", linestyle="--")
plt.title("SelectorAgent: Next State Prediction")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 5. Visualize thought bank in PCA space
with torch.no_grad():
    thoughts = agent.thought_bank.cpu().numpy()

pca = PCA(n_components=2)
thoughts_2d = pca.fit_transform(thoughts)

plt.figure(figsize=(7, 6))
plt.scatter(thoughts_2d[:, 0], thoughts_2d[:, 1], s=100, c=range(thoughts.shape[0]), cmap="plasma")
for i, (x, y) in enumerate(thoughts_2d):
    plt.text(x, y, str(i), fontsize=10, ha="center", va="center", color="white", weight="bold")
plt.title("Thought Bank (Latent Vectors) Projected to 2D")
plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Visualize training loss
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Training Loss over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
