import gym
import torch
import numpy as np


class EpsilonGreedyMulti:
    '''An Epsilon-Greedy policy for "d" stocks, with a model predicting actions for all stocks collectively.'''

    def __init__(self, epsilon_start: float, epsilon_min: float, epsilon_decay: float, env: gym.Env, model: torch.nn.Module):
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.model = model

    def __call__(self, states: np.ndarray):
        '''Select actions for the given states using the epsilon-greedy policy, for a model predicting collectively.'''
        coin = np.random.uniform()
        is_random_choice = coin < self.epsilon
        

        if is_random_choice:
            # Random action for each stock
            actions = np.random.randint(0, 3, size=self.env.d)
             # Assuming 3 actions (buy, sell, hold
            props = np.full(self.env.d, 0.05)  # Arbitrary probability for random actions (will be used to buy each stock with action > 0 in the same quantity)
        else:
            # Model prediction
            transformed_states, length_of_sequence = transform_state_sequence(states, self.env.T, self.env.d)
            normalized_close_prices = np.array(transformed_states[0]).reshape(1, length_of_sequence, self.env.d)
            account_balances = transformed_states[1]
            shares_held = np.array(transformed_states[2])
            normalized_close_prices_tensor = torch.tensor(normalized_close_prices, dtype=torch.float32, device=self.model.device)
            account_balances_tensor = torch.tensor(account_balances, dtype=torch.float32, device=self.model.device)
            shares_held_tensor = torch.tensor(shares_held, dtype=torch.float32, device=self.model.device)
            actions_tensor = self.model(normalized_close_prices_tensor, account_balances_tensor, shares_held_tensor).detach().cpu().numpy() 
            
            # Assuming the output is already in the desired format
            actions = np.argmax(actions_tensor, axis=2) 
            actions = actions.squeeze()
            # Choosing the best action for each stock
            
            props = [] # props will be the list of expected improvement of expected gain by choosing the best action over the second best action
            actions_tensor = actions_tensor.squeeze()
            for action_probs in actions_tensor:
                sorted_probs = np.sort(action_probs)[::-1]
                if sorted_probs[0] == 0:
                    props.append(0)
                else:
                    props.append((sorted_probs[0] - sorted_probs[1])/sorted_probs[0])
            #props will be used to buy (for the stock with action > 0) in  quantity proportional to the expected improvement of expected gain by choosing the best action over the second best action

        return actions.tolist(), props, is_random_choice

    def decay_epsilon(self):
        '''Decay the epsilon value after each episode.'''
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def transform_state_sequence(states, T, d):
    """
    Adjusts the state sequences to align with the DQN model's expected input format.
    
    Args:
        states: A tuple containing three elements: 
                - An array of price sequences for all stocks and all timesteps in the batch,
                - An array of account balances,
                - An array of shares held.
        T: The look-back period, or sequence length.
        d: The number of stocks.
    
    Returns:
        A numpy array of transformed states suitable for the DQN model input.
    """
    # Unpack the tuple into its components
    state_prices, state_balances, state_shares_held = states
    
    length_of_sequence = len(state_prices[0]) // d

    # Initialize a list to hold the transformed states
    

    # Iterate through each element in the batch
    new_state_format = []
    normalised_close_prices = []
    for t in range(length_of_sequence):
        for s in range(d):
            # Append the price for stock s at time t
            normalised_close_prices.append(state_prices[s][t])
        
        # Add the account balance and shares held to the end of the state, as per original format
    new_state_format.append(normalised_close_prices)
    new_state_format.append(state_balances)  # Account balance
    new_state_format.append(state_shares_held)  # Shares held for each stock (if applicable)
        
  
    return new_state_format, length_of_sequence
