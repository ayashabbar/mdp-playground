# MDP Playground: Bandits & Bayes-Adaptive Reinforcement Learning

[![Live Interactive Webpage](https://shields.io)](https://github.io)

An interactive Python and Jupyter Notebook playground exploring sequential decision-making under uncertainty—bridging classical **Multi-Armed Bandits**, **Bayesian Index Policies**, and **Bayes-Adaptive Markov Decision Processes (BAMDPs)**.

> **Read the full rendered tutorial, equations, and simulation plots online:**  
> **[https://github.io](https://github.io)**

---

## Key Topics & Implementations

### 1. Multi-Armed Bandits (`MDP/`)
* **Upper Confidence Bound (UCB):** Deterministic optimism-in-the-face-of-uncertainty action selection.
* **Thompson Sampling:** Bayesian probability matching using conjugate Beta-Bernoulli posterior updates and density evolution plots.
* **Gittins Indices:** Decomposition of multi-armed bandit problems via optimal stopping ("fair tax") values and asymptotic \(\gamma\)-quantile approximations.

### 2. Bayes-Adaptive MDPs (`BAMDP/`)
* **Mouse Foraging Environment:** A 4-state (`Safe`, `Dangerous`, `Predator`, `Death`) stochastic environment where the predator encounter probability \(\theta\) is initially unknown.
* **Exact Belief-State Tree Search (`Environment_Tree.py`):** Forward belief-tree expansion and backward dynamic programming over discretized posterior grids.
* **Policy Rollouts & Belief Tracking (`Environment.py`):** Monte-Carlo episode simulations tracking discounted return distributions, adaptive state trajectories, 90% credible intervals over \(\theta\), and cross-\(\theta\) robustness sweeps.

---

## Repository Structure

```text
mdp-playground/
├── BAMDP/
│   ├── Environment_Tree.py    # Exact forward/backward BAMDP tree search solver
│   └── Environment.py         # Mouse foraging rollout simulator & plotting utilities
├── MDP/                       # Multi-armed bandit algorithms (UCB, Thompson Sampling, Gittins)
├── html/                      # Static HTML export hosted via GitHub Pages
├── outputs/                   # Generated simulation figures and diagnostic plots
├── MDP_Playground.ipynb       # Main notebook combining theory, equations, code & plots
└── requirements.txt           # Python dependencies
```

---

## Getting Started

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com
cd mdp-playground
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `MDP_Playground.ipynb` in VS Code or Jupyter Lab:
```bash
jupyter lab MDP_Playground.ipynb
```

### 3. Regenerate the Static HTML Page
After updating `MDP_Playground.ipynb`, export the static webpage cleanly with:
```bash
quarto render MDP_Playground.ipynb --to html --output-dir html
# Or using nbconvert:
# jupyter nbconvert --to html --no-input --embed-images MDP_Playground.ipynb --output html/index.html
```

---

## References

1. **Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).** *A Tutorial on Thompson Sampling.* Foundations and Trends in Machine Learning, 11(1), 1–96. [[arXiv:1707.02038]](https://arxiv.org)
2. **Russo, D. (2019).** *Approximating Gittins Indices for Bayesian Bandits.* [[arXiv:1904.04732]](https://arxiv.org)
3. **Guez, A., Silver, D., & Dayan, P. (2013).** *Scalable and Efficient Bayes-Adaptive Reinforcement Learning Based on Monte-Carlo Tree Search.* Journal of Artificial Intelligence Research (JAIR), 48, 841–883. [[JAIR]](https://jair.org)
4. **Guez, A. (2015).** *Sample-Based Search Methods for Bayes-Adaptive Planning.* PhD Thesis, Gatsby Computational Neuroscience Unit, University College London (UCL).
