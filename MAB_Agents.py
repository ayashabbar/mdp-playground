from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from tqdm import trange

from environment import (
    BanditEnvironment,
    BernoulliBanditEnvironment,
    GaussianBanditEnvironment,
)


class Base_Bandit(ABC):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size

    def __init__(
        self,
        env: BanditEnvironment,
        epsilon=0.0,
        initial=0.0,
        step_size=0.1,
        sample_averages=False,
    ):
        self.env = env
        self.k = env.num_actions
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.epsilon = epsilon
        self.initial = initial

        self.average_reward = 0
        self.time = 0
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

    def reset(self):
        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        # # of chosen times for each action
        self.action_count = np.zeros(self.k)
        self.time = 0
        self.average_reward = 0

    # get an action for this bandit
    @abstractmethod
    def act(self) -> int:
        pass

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = self.env.get_reward(action)
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (
                reward - self.q_estimation[action]
            ) / self.action_count[action]
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (
                reward - self.q_estimation[action]
            )

        return reward

class EpsilonGreedy(Base_Bandit):
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])


class UCB(Base_Bandit):
    def __init__(self, UCB_param=2, **kwargs):
        super().__init__(**kwargs)
        self.UCB_param = UCB_param

    def act(self) -> int:
        UCB_estimation = self.q_estimation + self.UCB_param * np.sqrt(
            np.log(self.time + 1) / (self.action_count + 1e-5)
        )
        q_best = np.max(UCB_estimation)
        return np.random.choice(np.where(UCB_estimation == q_best)[0])


class ThompsonSampling(Base_Bandit):
    def __init__(
        self, env: BanditEnvironment, default_alpha=1, default_beta=1, **kwargs
    ):
        if not isinstance(env, BernoulliBanditEnvironment):
            raise TypeError(
                "ThompsonSampling requires a BernoulliBanditEnvironment instance."
            )
        # Thompson Sampling specific parameters
        self.default_alpha = default_alpha
        self.default_beta = default_beta

        # Call the superclass constructor with all remaining parameters
        super().__init__(env=env, **kwargs)

        # Initialize Thompson Sampling specific attributes using k from superclass
        self.alphas = [self.default_alpha] * self.k
        self.betas = [self.default_beta] * self.k
        self.successes = [0] * self.k
        self.failures = [0] * self.k

    def reset(self):
        self.alphas = [self.default_alpha] * self.k
        self.betas = [self.default_beta] * self.k
        self.successes = [0] * self.k
        self.failures = [0] * self.k

    def act(self) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return int(np.argmax(samples))

    def step(self, action: int):
        """
        Simulates the environment: generates a reward based on true probabilities,
        updates the internal counts, and returns the reward.
        """
        reward = self.env.get_reward(action)
        self.time += 1
        self.average_reward += (reward - self.average_reward) / self.time

        # update internal statistics
        if reward == 1:
            self.successes[action] += 1
            self.alphas[action] += 1
        else:
            self.failures[action] += 1
            self.betas[action] += 1

        return reward

    def update(self, arm, reward):
        pass

    def plot_posteriors(self, arm_labels=None):
        # plots the Probability Density Function (PDF) of the Beta distribution for each arm."""
        n_arms = len(self.alphas)
        if arm_labels is None:
            arm_labels = [f"Arm {i + 1}" for i in range(n_arms)]

        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(10, 6))

        for i in range(n_arms):
            # calculate shape parameters: prior + observed
            a = self.alphas[i]  # equivalent to successes + prior_alpha
            b = self.betas[i]  # equivalent to failures + prior_beta

            pdf = beta.pdf(x, a, b)

            label = f"{arm_labels[i]} ($\\alpha$={a}, $\\beta$={b})"
            if i == self.env.best_action:
                plt.plot(x, pdf, label=label, linewidth=2.5, linestyle="--")
            else:
                plt.plot(x, pdf, label=label)

        plt.xlabel("Probability of Success ($\\theta$)")
        plt.ylabel("Probability Density")
        plt.title("Posterior Beta Distributions for Each Arm")
        plt.legend()
        # plt.show()


class GittinsIndex(Base_Bandit):
    def __init__(self, discount_factor=0.99, **kwargs
    ):
        self.discount_factor = discount_factor
        super().__init__(**kwargs)

        # Initialize flat Beta(1,1) priors for each arm
        self.successes = np.ones(self.k)
        self.failures = np.ones(self.k)

    def act(self) -> int:
        """Finds and returns the index of the best arm to pull."""
        indices = []
        for i in range(self.k):
            idx = gittins_index_approx(
                self.successes[i], 
                self.failures[i], 
                self.discount_factor
            )
            indices.append(idx)
        return int(np.argmax(indices))

    def update(self, arm_index, reward):
        """Updates the posterior state based on binary outcomes."""
        if reward == 1:
            self.successes[arm_index] += 1
        else:
             self.failures[arm_index] += 1    


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.env.reset()
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.env.best_action:
                    best_action_counts[i, r, t] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_epsilongreedy(runs=2000, time=1000, show=False):
    env = GaussianBanditEnvironment(num_actions=10)
    epsilons = [0, 0.1, 0.01]
    bandits = [
        EpsilonGreedy(env=env, epsilon=eps, sample_averages=True) for eps in epsilons
    ]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label=r"$\epsilon = %.02f$" % (eps))
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label=r"$\epsilon = %.02f$" % (eps))
    plt.xlabel("steps")
    plt.ylabel("optimal action")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig("figure_epsilongreedy.png")
    plt.close()


def figure_UCB(runs=2000, time=1000, show=False):
    env = GaussianBanditEnvironment(num_actions=10)
    bandits = []
    bandits.append(UCB(env=env, epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(EpsilonGreedy(env=env, epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label="UCB $c = 2$")
    plt.plot(average_rewards[1], label="epsilon greedy $\\epsilon = 0.1$")
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig("figure_UCB.png")
    plt.close()


def figure_TS(runs=2000, time=1000, show=False):
    true_probs = [0.2, 0.7, 0.5]
    env = BernoulliBanditEnvironment(true_probs=true_probs)

    ts = ThompsonSampling(env=env)
    bandits_list = [ts]

    rewards, optimality = simulate(runs, time, bandits=bandits_list)
    avg_rewards = rewards[0]
    avg_optimality = optimality[0]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards)
    plt.title("Average Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(avg_optimality * 100)
    plt.title("Percentage of Optimal Actions")
    plt.xlabel("Step")
    plt.ylabel("% Optimal")
    plt.ylim(0, 100)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("figure_TS.png")
    plt.close()

    # visualize final posteriors
    ts_viz = ThompsonSampling(env=env)

    for _ in range(200):
        action = ts_viz.act()
        reward = ts_viz.step(action)

    ts_viz.plot_posteriors(arm_labels=["Low (0.2)", "High (0.7)", "Med (0.5)"])
    plt.savefig("figure_TS_PDF.png")
    plt.close()


def gittins_index_approx(successes: int, failures: int, discount_factor: float = 0.9) -> float:
    """
    This function implements a Bayesian Upper Confidence Bound (Bayes-UCB) strategy 
    that serves as a computationally efficient proxy for the true Gittins index
    Daniel Russo (2021)

    Parameters:
        successes (int): Number of observed successes (s).
        failures (int): Number of observed failures (f).
        discount_factor (float): Gamma, typically close to 1 (e.g., 0.9 - 0.99).
        
    Returns:
        float: Approximated Gittins index.
    """
    # Alpha and Beta for the posterior Beta distribution
    # Prior is typically Beta(1, 1) for uniform, so alpha = s + 1, beta_param = f + 1
    alpha = successes + 1
    beta_param = failures + 1
    
    n = alpha + beta_param - 2 # Total observations
    
    # Posterior Mean (Exploitation term)
    mean = alpha / (alpha + beta_param)
    
    # If no data, return the prior mean (0.5 for Beta(1,1))
    if n == 0:
        return 0.5

    # Posterior Variance (Exploration term) / Uncertainty
    variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
    std_dev = np.sqrt(variance)
    
    # Effective horizon approximation
    if discount_factor >= 1.0:
        return 1.0 
        
    effective_horizon = 1 / (1 - discount_factor)
    
    # The exploration bonus (the approximation core)
    scaling_factor = np.sqrt(2 * np.log(effective_horizon + 1)) 
    
    exploration_bonus = scaling_factor * std_dev
    
    index_value = mean + exploration_bonus
    
    return float(min(max(index_value, 0.0), 1.0))


def figure_GI():
    # Setup simulation parameters
    fixed_failures = 5
    success_range = np.arange(1, 21)  # X-axis from 1 to 20 successes
    discount = 0.95

    # Compute indices over the distribution grid
    gittins_values = [gittins_index_approx(s, fixed_failures, discount) for s in success_range]
    posterior_means = [s / (s + fixed_failures) for s in success_range]

    # Initialize the visualization layout
    plt.figure(figsize=(10, 6), facecolor='#f8f9fa')
    ax = plt.axes()
    ax.set_facecolor('#ffffff')

    # Plot the curves
    plt.plot(success_range, gittins_values, color='#1a73e8', marker='o', 
            linestyle='-', linewidth=2.5, label='Gittins Index (Discount=0.95)')
    plt.plot(success_range, posterior_means, color='#d93025', marker='x', 
            linestyle='--', linewidth=2.0, label='Posterior Mean (Exploitation Only)')

    # Fill the gap to explicitly highlight the exploration bonus gap
    plt.fill_between(success_range, posterior_means, gittins_values, 
                    color='#1a73e8', alpha=0.1, label='Exploration Bonus')

    plt.title('Gittins Index Optimization Surface Map', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Observed Successes (\\$alpha - 1$)', fontsize=12, labelpad=10)
    plt.ylabel('Policy Value / Allocation Score', fontsize=12, labelpad=10)
    plt.xticks(success_range)
    plt.ylim(0.0, 1.05)

    #plt.grid(True, linestyle=':', alpha=0.6, color='#9aa0a6')
    plt.legend(fontsize=11, loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#dadce0')

    plt.tight_layout()
    plt.savefig("figure_GI.png")
    plt.close()


if __name__ == "__main__":
    #figure_epsilongreedy()
    #figure_UCB()
    #figure_TS()
    figure_GI()
