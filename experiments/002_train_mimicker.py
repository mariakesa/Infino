
import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.mimicker import MimickerAgent
import numpy as np

# 1. Simulate the world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=200)
x = np.array(trajectory["x"])

# 2. Build dataset: input = x_t, target = x_{t+1}
x_input = x[:-1]
x_target = x[1:]

# 3. Train the Mimicker
agent = MimickerAgent(hidden_dim=32)
loss_history = agent.train_on_data(x_input, x_target, epochs=300, lr=0.01)

# 4. Predict on training input
x_pred = agent.predict(x_input)

# 5. Plot predictions vs ground truth
plt.figure(figsize=(10, 4))
plt.plot(x_input, x_target, label="True xₜ₊₁", alpha=0.7)
plt.plot(x_input, x_pred, label="Predicted xₜ₊₁", linestyle='dashed')
plt.xlabel("xₜ")
plt.ylabel("xₜ₊₁")
plt.title("Mimicker Agent Prediction")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 6. Plot training loss
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.show()

plt.plot(x_pred, label="Predicted xₜ₊₁")
plt.show()
