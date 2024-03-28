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


    def __call__(self, state: np.ndarray, type_of_model: str):
        '''Select an action for the given state using the epsilon-greedy policy.'''
        coin = np.random.uniform()
        is_random_choice = True

        if coin < self.epsilon:
            action = np.random.randint(0, 3)
            prop = 0.05
            
        else:
            is_random_choice = False
            normalized_close_price, account_balance, shares_held = state
            normalized_close_price = np.array(normalized_close_price)
            # from shape (21,) to (1,21,1)
            normalized_close_price = normalized_close_price.reshape(1, normalized_close_price.shape[0],1)
            normalized_close_price_tensor = torch.tensor(normalized_close_price, dtype=torch.float, device=self.model.device)
            account_balance_tensor = torch.tensor(account_balance, dtype=torch.float, device=self.model.device)
            shares_held_tensor = torch.tensor(shares_held, dtype=torch.float, device=self.model.device)

            sequence_length = normalized_close_price_tensor.size(1)

            if type_of_model == "DQN" : 
                actions = self.model(normalized_close_price_tensor,
                                    account_balance_tensor, 
                                    shares_held_tensor
                                    ).detach().cpu().numpy().squeeze()

            if type_of_model == 'DQN_with_Transformer':

                tgt_mask = self.model.get_tgt_mask(sequence_length).to(self.model.device)

                actions = self.model(normalized_close_price_tensor,
                                    account_balance_tensor, 
                                    shares_held_tensor,
                                    tgt_mask
                                    ).detach().cpu().numpy().squeeze()
                
            action = np.argmax(actions)
            sorted_arr = np.sort(actions)[::-1]
            prop = sorted_arr[0] - sorted_arr[1]
         

        return action, prop, is_random_choice


    def decay_epsilon(self):
        '''Decay the epsilon value after each episode.'''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)