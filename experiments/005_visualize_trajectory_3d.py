
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import numpy as np

from infino.world.bouncing_ball import BouncingBallWorld
from infino.agents.mimicker import MimickerAgent

def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# 1. Simulate data
world = BouncingBallWorld()
trajectory = world.simulate(n_steps=200)
x = trajectory["x"][:-1]
y = trajectory["x"][1:]

x_tensor = torch.FloatTensor(x).unsqueeze(1)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# 2. Initialize model and optimizer
model = MimickerAgent()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

param_history = []
loss_history = []

# 3. Training with trajectory tracking
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()

    # Save flattened parameters every epoch
    param_vector = flatten_params(model).detach().cpu().numpy()
    param_history.append(param_vector)
    loss_history.append(loss.item())

# 4. Convert param trajectory to 3D using PCA
param_matrix = np.stack(param_history)  # shape: (epochs, num_params)
pca = PCA(n_components=3)
trajectory_3d = pca.fit_transform(param_matrix)

# 5. 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], marker='o', linewidth=1, markersize=2)
ax.set_title("3D Optimization Trajectory in Parameter Space (PCA-Projected)")
ax.set_xlabel("PCA Dim 1")
ax.set_ylabel("PCA Dim 2")
ax.set_zlabel("PCA Dim 3")
plt.tight_layout()
#plt.show()

print(trajectory_3d)