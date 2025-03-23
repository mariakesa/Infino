
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from minisom import MiniSom

# Load trajectory
trajectory = np.load("pl2ku_trajectory.npy")

# Train SOM
som = MiniSom(x=2, y=2, input_len=trajectory.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(trajectory)
som.train(trajectory, 1000, verbose=True)

# Assign each point to a SOM unit
assignments = np.array([som.winner(x) for x in trajectory])
unit_ids = np.array([x * som._weights.shape[1] + y for x, y in assignments])  # flatten 2D coords

# PCA for visualization
pca = PCA(n_components=3)
trajectory_pca = pca.fit_transform(trajectory)

# Plot PCA trajectory colored by SOM unit ID
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(trajectory_pca[:, 0], trajectory_pca[:, 1], trajectory_pca[:, 2],
                c=unit_ids, cmap='tab20', s=10)
ax.set_title("Learning Trajectory Colored by SOM Unit")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
plt.colorbar(sc, label='SOM Unit ID')
plt.tight_layout()
plt.show()
