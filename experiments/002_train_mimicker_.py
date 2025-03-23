
import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.mimicker import MimickerAgent

# 1. Simulate world
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=200)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

# 2. Initialize Mimicker Agent
agent = MimickerAgent()

# 3. Train
losses = agent.train_on_data(x, y, epochs=300, lr=0.01)

# 4. Predict
predictions = agent.predict(x)

# 5. Plot predictions vs true values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, label='True xₜ₊₁')
plt.plot(x, predictions, label='Predicted xₜ₊₁')
plt.title("Prediction vs True")
plt.xlabel("xₜ")
plt.ylabel("xₜ₊₁")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.tight_layout()
plt.show()
