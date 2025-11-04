import numpy as np
import matplotlib.pyplot as plt
from infino.agents.selector_agent_curious_bn import CuriousSelectorAgent

# === Multi-frequency Sinewave Dataset ===
def generate_multi_sine_dataset(num_samples=200, freq_range=(1, 5), noise_std=0.1):
    x = np.linspace(0, 1, num_samples)
    freqs = np.random.uniform(freq_range[0], freq_range[1], size=num_samples)
    y = np.sin(2 * np.pi * freqs * x) + np.random.normal(0, noise_std, size=num_samples)
    return np.stack([x, freqs], axis=1), y

# === Train and Plot ===
def train_and_plot():
    # Generate data
    X, y = generate_multi_sine_dataset()

    # Create agent
    agent = CuriousSelectorAgent(input_dim=2, latent_dim=16, num_thoughts=12)
    agent.train_on_data(X, y, epochs=300)

    # Predict
    y_pred = agent.predict(X)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(X[:, 0], y, label='Ground Truth', alpha=0.6)
    plt.plot(X[:, 0], y_pred, label='Prediction', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Curious Agent on Multi-Frequency Sine Task")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_and_plot()