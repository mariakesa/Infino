import torch
from torch.utils.data import Dataset
import numpy as np

class SineWaveTaskDataset(Dataset):
    def __init__(self, num_tasks=100, num_points=50, noise_std=0.1):
        super().__init__()
        self.data = []

        for _ in range(num_tasks):
            A = np.random.uniform(0.1, 5.0)
            f = np.random.uniform(0.5, 2.0)
            phi = np.random.uniform(0, 2 * np.pi)

            x = np.linspace(-5, 5, num_points)
            y = A * np.sin(f * x + phi) + np.random.normal(0, noise_std, size=x.shape)

            self.data.append((x.astype(np.float32), y.astype(np.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    X = [torch.from_numpy(item[0]) for item in batch]
    Y = [torch.from_numpy(item[1]) for item in batch]
    return torch.stack(X), torch.stack(Y)

import numpy as np

def generate_sinewave_dataset(num_samples=256, noise_std=0.1):
    x = np.linspace(0, 1, num_samples)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_std, size=num_samples)
    return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = SineWaveTaskDataset(num_tasks=3)
    x, y = dataset[0]
    plt.plot(x, y)
    plt.title("Example Sine Wave Task")
    plt.show()
