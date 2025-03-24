
import torch
import numpy as np
import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.selector_agent_curious import CuriousSelectorAgent
import torch.nn.functional as F

# 1. Simulate world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=300)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# 2. Initialize curious agent
agent = CuriousSelectorAgent(
    input_dim=1,
    latent_dim=8,
    num_thoughts=12,
    temperature=0.3,
    hard=True
)

# 3. Train agent
losses = agent.train_on_data(x, y, epochs=300, lr=0.01)
y_pred = agent.predict(x)

# 4. Get selector logits and selected thought IDs
x_tensor = torch.FloatTensor(x).unsqueeze(1)
with torch.no_grad():
    logits = agent.selector(x_tensor)
    gumbel_weights = F.gumbel_softmax(logits, tau=agent.temperature, hard=True, dim=-1)
    selected_ids = gumbel_weights.argmax(dim=-1).cpu().numpy()

# 5. Plot predictions colored by chosen thoughts
plt.figure(figsize=(12, 5))
for t in range(len(y_pred)):
    plt.plot([t, t+1], [y_pred[t], y_pred[t]], color=plt.cm.tab20(selected_ids[t] % 20))
plt.plot(y, label="True $x_{t+1}$", color="black", alpha=0.6, linewidth=1)
plt.title("CuriousSelectorAgent Predictions Colored by Chosen Thought")
plt.xlabel("Time Step")
plt.ylabel("Predicted Position")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Plot thought ID usage over time
plt.figure(figsize=(10, 3))
plt.plot(selected_ids, drawstyle="steps-post", color="green")
plt.title("Curious Thought ID Over Time")
plt.xlabel("Time Step")
plt.ylabel("Thought ID")
plt.yticks(np.arange(agent.num_thoughts))
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Training Loss (CuriousSelectorAgent)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Print thought usage frequency
counts = np.bincount(selected_ids, minlength=agent.num_thoughts)
for i, c in enumerate(counts):
    print(f"Thought {i}: used {c} times")
