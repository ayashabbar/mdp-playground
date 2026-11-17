import random
from abc import ABC, abstractmethod

import numpy as np


class BanditEnvironment(ABC):
    """Abstract base class for bandit environments"""

    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._best_action = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_reward(self, action) -> float:
        pass

    @property
    def best_action(self):
        return self._best_action

    @property
    def num_actions(self):
        return self._num_actions


class GaussianBanditEnvironment(BanditEnvironment):
    def __init__(self, num_actions: int = 10, true_reward: float = 0.0):
        super().__init__(num_actions)
        self.true_reward = true_reward
        self.q_true = np.random.randn(self.num_actions) + self.true_reward
        self._best_action = np.argmax(self.q_true)

    def reset(self):
        self.q_true = np.random.randn(self.num_actions) + self.true_reward
        self._best_action = np.argmax(self.q_true)

    def get_reward(self, action) -> float:
        return np.random.randn() + self.q_true[action]


class BernoulliBanditEnvironment(BanditEnvironment):
    def __init__(self, true_probs: list):
        super().__init__(len(true_probs))
        self.true_probs = np.array(true_probs)
        self._best_action = np.argmax(true_probs)

    def reset(self):
        pass

    def get_reward(self, action):
        return 1 if random.random() < self.true_probs[action] else 0
