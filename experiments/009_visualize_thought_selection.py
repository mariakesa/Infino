
import torch
import numpy as np
import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.selector_agent_gumbel import SelectorAgent
import torch.nn.functional as F

# 1. Simulate bouncing ball
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=300)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# 2. Initialize agent
agent = SelectorAgent(
    input_dim=1,
    latent_dim=8,
    num_thoughts=12,
    temperature=0.2,
    hard=True
)
agent.train_on_data(x, y, epochs=300, lr=0.01)

# 3. Get selector logits & predicted thought IDs
x_tensor = torch.FloatTensor(x).unsqueeze(1)
with torch.no_grad():
    logits = agent.selector(x_tensor)  # (T, num_thoughts)
    thought_probs = F.gumbel_softmax(logits, tau=agent.temperature, hard=True, dim=-1)
    selected_ids = thought_probs.argmax(dim=-1).cpu().numpy()  # (T,)
    y_pred = agent.predict(x)

# 4. Plot prediction colored by selected thought
plt.figure(figsize=(12, 5))
for t in range(len(y_pred)):
    plt.plot([t, t + 1], [y_pred[t], y_pred[t]], color=plt.cm.tab20(selected_ids[t] % 20))
plt.plot(y, label="True $x_{t+1}$", color="black", linewidth=1, alpha=0.6)
plt.title("SelectorAgent Predictions Colored by Chosen Thought")
plt.xlabel("Time Step")
plt.ylabel("Predicted Position")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Plot selected thought IDs over time
plt.figure(figsize=(10, 3))
plt.plot(selected_ids, drawstyle="steps-post", color="purple")
plt.title("Selected Thought ID Over Time")
plt.xlabel("Time Step")
plt.ylabel("Thought ID")
plt.yticks(np.arange(agent.num_thoughts))
plt.grid(True)
plt.tight_layout()
plt.show()
