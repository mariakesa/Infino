import torch
import matplotlib.pyplot as plt
from infino.agents.selector_agent_curious_bn import CuriousSelectorAgent
from infino.world.sinewave_task import generate_sinewave_dataset

# 1. Hyperparameters
input_dim = 1         # Scalar x input
latent_dim = 8        # Size of each thought vector
num_thoughts = 16     # Number of thoughts in the bank
temperature = 0.5
epochs = 500
lr = 0.01

# 2. Generate data
x, y = generate_sinewave_dataset(num_samples=256, noise_std=0.05)

# 3. Initialize agent
agent = CuriousSelectorAgent(
    input_dim=input_dim,
    latent_dim=latent_dim,
    num_thoughts=num_thoughts,
    temperature=temperature,
    hard=True
)

# 4. Train
loss_history = agent.train_on_data(x, y, epochs=epochs, lr=lr)

# 5. Plot loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Curious Agent Training Loss")
plt.show()

# 6. Predict on test points
import numpy as np
x_test = np.linspace(0, 1, 256)
y_pred = agent.predict(x_test)

# 7. Plot predictions
plt.figure(figsize=(10, 4))
plt.plot(x_test, y_pred, label='Prediction', color='royalblue')
plt.plot(x, y, label='Ground Truth', color='orange', alpha=0.5)
plt.legend()
plt.title("Curious Agent Predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
