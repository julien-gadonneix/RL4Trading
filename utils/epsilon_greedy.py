import gym
import torch
import numpy as np


class EpsilonGreedy:
    '''An Epsilon-Greedy policy.'''

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 env: gym.Env,
                 model: torch.nn.Module):
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.model = model


    def __call__(self, state: np.ndarray) -> np.int64:
        '''Select an action for the given state using the epsilon-greedy policy.'''
        coin = np.random.uniform()

        if coin < self.epsilon:
            action = np.random.uniform(-1, 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            action = self.model(state_tensor).detach().cpu().numpy().item()

        return action


    def decay_epsilon(self):
        '''Decay the epsilon value after each episode.'''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)