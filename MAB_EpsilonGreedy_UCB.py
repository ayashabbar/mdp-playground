import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import random
from numpy.random import beta, binomial
from scipy.stats import beta

from abc import ABC, abstractmethod

class Base_Bandit(ABC):
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action

    def __init__(
        self,
        k_arm=10,
        epsilon=0.0,
        initial=0.0,
        step_size=0.1,
        sample_averages=False,
        UCB_param=None,
        true_reward=0.0,
    ):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0


    # get an action for this bandit
    @abstractmethod
    def act(self):
        pass

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
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
    def act(self):
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
            self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])    


class ThompsonSampling(Base_Bandit):
    def __init__(self, alphas=None, betas=None, successes=None, failures=None):
        self.k_arm = 0
        # default to uniform priors (1, 1) if not provided
        self.default_alpha = 1
        self.default_beta = 1
            
        # will be initialized in initialize() or reset()
        self.alphas = alphas
        self.betas = betas
        self.successes = successes
        self.failures = failures
        self.best_action = None
        self.true_probs = None

    def initialize(self, k_arm, true_probs=None):
        self.k_arm = k_arm
        self.true_probs = true_probs
        if true_probs:
            self.best_action = int(np.argmax(true_probs))
                
        # reset to uniform priors
        self.alphas = [self.default_alpha] * k_arm
        self.betas = [self.default_beta] * k_arm
        self.successes = [0] * k_arm
        self.failures = [0] * k_arm

    def reset(self):
        # resets the internal state for a new run
        if self.k_arm > 0:
            self.alphas = [self.default_alpha] * self.k_arm
            self.betas = [self.default_beta] * self.k_arm
            self.successes = [0] * self.k_arm
            self.failures = [0] * self.k_arm

    def act(self) -> int:
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return int(np.argmax(samples))

    def step(self, action: int) -> float:
        """
        Simulates the environment: generates a reward based on true probabilities,
        updates the internal counts, and returns the reward.
        """
        if self.true_probs is None:
            raise ValueError("true_probs must be set via initialize() before calling step()")
                
        # generate reward (Bernoulli trial)
        reward = 1 if random.random() < self.true_probs[action] else 0
            
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
            arm_labels = [f"Arm {i+1}" for i in range(n_arms)]
            
        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(10, 6))
        
        for i in range(n_arms):
            # calculate shape parameters: prior + observed
            a = self.alphas[i] # equivalent to successes + prior_alpha
            b = self.betas[i]  # equivalent to failures + prior_beta
            
            pdf = beta.pdf(x, a, b)
            
            label = f"{arm_labels[i]} ($\\alpha$={a}, $\\beta$={b})"
            if i == self.best_action:
                plt.plot(x, pdf, label=label, linewidth=2.5, linestyle='--')
            else:
                plt.plot(x, pdf, label=label)

        plt.xlabel("Probability of Success ($\\theta$)")
        plt.ylabel("Probability Density")
        plt.title("Posterior Beta Distributions for Each Arm")
        plt.legend()
        #plt.show()      


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1 

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_epsilongreedy(runs=2000, time=1000, show=False):
    epsilons = [0, 0.1, 0.01]
    bandits = [EpsilonGreedy(epsilon=eps, sample_averages=True) for eps in epsilons]
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
    bandits = []
    bandits.append(UCB(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(UCB(epsilon=0.1, sample_averages=True))
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
    k_arms = len(true_probs)

    ts = ThompsonSampling()
    ts.initialize(k_arm=k_arms, true_probs=true_probs)
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
    ts_viz = ThompsonSampling()
    ts_viz.initialize(k_arm=k_arms, true_probs=true_probs)

    for _ in range(200):
        action = ts_viz.act()
        reward = ts_viz.step(action)

    ts_viz.plot_posteriors(arm_labels=["Low (0.2)", "High (0.7)", "Med (0.5)"])
    plt.savefig("figure_TS_PDF.png")
    plt.close()


 
if __name__ == "__main__":
    figure_epsilongreedy()
    figure_UCB()
    figure_TS()