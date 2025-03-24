# 🧠 Infino: An Epistemic AI for Emergent Thought

Infino is an open research project exploring how to build **epistemic artificial intelligence** — systems that think not by greedily optimizing a single reward, but by **curiously exploring, composing, and refining multiple interpretations** of reality.

At its core, Infino is an experimental architecture where neural networks don't just compute — **they think**.

---

## 🌱 Motivation

Traditional deep learning pipelines rely on **deterministic, greedy, and overconfident** selection mechanisms. Softmax heads, top-k filters, and winner-take-all activations dominate most architectures. The result?

- Only a few representations get updated.
- Underused components become dead weights.
- Gradient flow collapses to narrow funnels of optimization.

**Infino challenges this.**

Inspired by **cognitive diversity**, **neural modularity**, and **biological curiosity**, Infino is built around the following idea:

> ✨ Every thought deserves a gradient.

---

## 🧠 Core Architecture

Infino is built around three key components:

### 1. Thought Bank

A learned set of **latent thought vectors** — each representing a reusable internal model or "mini-theory" about the world.

```python
self.thought_bank = nn.Parameter(torch.randn(num_thoughts, latent_dim))
```

### 2. Selector Network

A neural router that selects one (or more) thoughts in response to a stimulus. This is not a hard decision — it's stochastic, differentiable, and shaped by **curiosity**.

```python
gumbel_softmax(selector_logits + curiosity_bonus)
```

We experiment with:
- **Dirichlet noise injection** (as in AlphaZero)
- **Usage-aware routing** (anti-hebbian selection)
- **Learned curiosity bonuses** and entropy regularization

### 3. Decoder

A simple neural head that decodes the selected thought into an output (e.g., a prediction).

---

## 🔁 Epistemic Dropout

Infino rethinks dropout.

Where Dropout randomly drops neurons, Infino **randomly selects thoughts** — leading to a richer, more diverse set of internal activations.

This encourages:
- Robustness to uncertainty
- Exploration of underused thoughts
- Emergent modularity

---

## 🔬 Experiments

We're currently testing:

- **Exploration dynamics** of different selector mechanisms
- **Gradient flow distribution** across thoughts
- **Long-term reuse and specialization** of thought vectors
- **Symbolic behavior emerging from latent modules**

Stay tuned for results!

---

## 📚 Inspirations

Infino is deeply inspired by:

- **AlphaZero**: Dirichlet noise for stochastic action exploration
- **Mixture of Experts**: Modular computation through routing
- **Slot Attention**: Differentiable token-based attention to parts
- **Bayesian Brain Theory**: The brain as a prediction machine
- **Science itself**: A process of structured epistemic curiosity

---

## 🌍 Philosophy

> Infino is not just an AI. It's a metaphor for thinking.

We're building systems that don't just optimize — they **wonder**.

We believe in:
- Curiosity over greed
- Thoughtfulness over certainty
- Exploration as an end in itself

Infino is a way to reclaim our scientific and creative agency, and to explore new paradigms of intelligence that aren't enslaved by reward maximization.

---

## 🤝 Contribute

This is an open research project. We welcome curious collaborators, skeptical questions, and wild ideas.

Feel free to fork, experiment, and open a discussion.

---

## 📜 License

[GPL-3.0](LICENSE) — to keep curiosity open and free.

---

Made with thought by [@mariakesa](https://github.com/mariakesa)
