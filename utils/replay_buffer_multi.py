import collections
import numpy as np
from typing import Tuple
import random

class ReplayBufferMulti:
    """A Replay Buffer for environments with multiple stocks."""

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: Tuple[list, float, np.ndarray], action: np.ndarray, reward: float, next_state: Tuple[list, float, np.ndarray], done: bool):
        '''Add a set to the buffer. The state and next_state are now tuples containing observation details for multiple stocks.'''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        '''Sample a batch from the buffer and return it in a format suitable for multi-stock observations.'''
        mini_batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batched observations, actions, rewards, next observations, and dones
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Separate the observation components for states and next_states
        state_prices, state_balances, state_shares_held = zip(*[(s[0], s[1], s[2]) for s in states])
        next_state_prices, next_state_balances, next_state_shares_held = zip(*[(s[0], s[1], s[2]) for s in next_states])

        # Convert lists of arrays/lists to numpy arrays for compatibility with machine learning frameworks
        state_prices = np.array(state_prices)
        state_balances = np.array(state_balances)
        state_shares_held = np.array(state_shares_held)
        
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_state_prices = np.array(next_state_prices)
        next_state_balances = np.array(next_state_balances)
        next_state_shares_held = np.array(next_state_shares_held)
        dones = np.array(dones)

        # Return the batched observations, actions, rewards, next observations, and dones in the correct format
        return (state_prices, state_balances, state_shares_held), actions, rewards, (next_state_prices, next_state_balances, next_state_shares_held), dones

    def __len__(self):
        '''Return the length of the buffer.'''
        return len(self.buffer)
       

