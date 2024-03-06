import collections
import numpy as np
from typing import Tuple
import random


class ReplayBuffer:
    """A Replay Buffer as in the exercise session."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)


    def add(self, state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool):
        '''Add a set to the buffer.'''
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]:
        '''Sample a batch from the buffer.'''
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones


    def __len__(self):
        '''Return the length of the buffer.'''
        return len(self.buffer)