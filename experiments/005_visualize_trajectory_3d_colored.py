
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

    param_vector = flatten_params(model).detach().cpu().numpy()
    param_history.append(param_vector)
    loss_history.append(loss.item())

# 4. PCA projection to 3D
param_matrix = np.stack(param_history)
print('BOOM! ', param_matrix.shape)
trajectory_3d = PCA(n_components=3).fit_transform(param_matrix)
loss_values = np.array(loss_history)

# 5. Normalize loss for color mapping
norm = plt.Normalize(loss_values.min(), loss_values.max())
colors = plt.cm.viridis(norm(loss_values))

# 6. Plot colored 3D trajectory
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(trajectory_3d) - 1):
    x_line = trajectory_3d[i:i+2, 0]
    y_line = trajectory_3d[i:i+2, 1]
    z_line = trajectory_3d[i:i+2, 2]
    ax.plot(x_line, y_line, z_line, color=colors[i], linewidth=2)

ax.set_title("3D Optimization Trajectory (Colored by Loss)")
ax.set_xlabel("PCA Dim 1")
ax.set_ylabel("PCA Dim 2")
ax.set_zlabel("PCA Dim 3")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Loss Value')

plt.tight_layout()
plt.show()

print(trajectory_3d)

import numpy as np

np.save('pl2ku_trajectory.npy', trajectory_3d)

