import numpy as np
import torch
from tqdm.notebook import tqdm
import itertools

class MultiStockAgent:
    '''Agent class to interact with the multi-stock trading environment.'''

    def __init__(self, env, model, target_model, target_q_network_sync_period, optimizer, lr_scheduler, loss_fn, replay_buffer, epsilon_greedy):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.target_q_network_sync_period = target_q_network_sync_period
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.replay_buffer = replay_buffer
        self.epsilon_greedy = epsilon_greedy
        # Adjusted for multi-stock
        self.num_stocks = env.d  
        self.action_list = list()
        self.wealth_list = list()
        self.random_choice_list = list()

    def train(self, num_episodes, gamma, batch_size):
        '''Trains the agent.'''
        iteration = 0
        episode_reward_list = []
        episode_reward = 0.

        for episode_index in tqdm(range(1, num_episodes+1)):
            self.action_list.append([])
            self.wealth_list.append([])
            self.random_choice_list.append([])
            # print(f"Episode {episode_index}")
            state = self.env.reset()
            episode_reward = 0.
            for iteration in itertools.count():

                action, prop, is_random_choice = self.epsilon_greedy(state)
                self.random_choice_list[-1].append(is_random_choice)
                self.action_list[-1].append(action)
                next_state, reward, done = self.env.step(action, prop)
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                self.wealth_list[-1].append(1000+episode_reward)

                actual_batch_size = min(len(self.replay_buffer), batch_size)
 

                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(actual_batch_size)

                # Transform both states and next_states to match the DQN model input requirement
                transformed_batch_states, length_of_sequence= transform_state_sequence(batch_states, self.env.T, self.env.d)
                transformed_batch_next_states,length_of_sequence = transform_state_sequence(batch_next_states, self.env.T, self.env.d)

                # Now, you can reshape and convert these transformed states to tensors as per the original code
                # Ensure the reshaping below matches the new structure (you might not need reshaping if your structure aligns directly)
                normalized_close_prices = np.array([state[0] for state in transformed_batch_states]).reshape(actual_batch_size, length_of_sequence, self.env.d)
                account_balances = np.array([state[1] for state in transformed_batch_states])
                shares_held = np.array([state[2] for state in transformed_batch_states])

                # Convert to tensors, similar to the original code but without unnecessary reshaping
                normalized_close_prices_tensor = torch.tensor(normalized_close_prices, dtype=torch.float32, device=self.model.device)
                account_balances_tensor = torch.tensor(account_balances, dtype=torch.float32, device=self.model.device)
                shares_held_tensor = torch.tensor(shares_held, dtype=torch.float32, device=self.model.device)

                # Repeat the above process for next states
                next_normalized_close_prices = np.array([state[0] for state in transformed_batch_next_states]).reshape(actual_batch_size, length_of_sequence, self.env.d)
                next_account_balances = np.array([state[1] for state in transformed_batch_next_states])
                next_shares_held = np.array([state[2] for state in transformed_batch_next_states])

                next_normalized_close_prices_tensor = torch.tensor(next_normalized_close_prices, dtype=torch.float32, device=self.model.device)
                next_account_balances_tensor = torch.tensor(next_account_balances, dtype=torch.float32, device=self.model.device)
                next_shares_held_tensor = torch.tensor(next_shares_held, dtype=torch.float32, device=self.model.device)

                # `batch_actions` is a list of lists where each inner list contains discretized actions for each stock
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.model.device)  # Shape: [batch_size, num_stocks]

                # Convert batch_rewards, batch_dones into tensors
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.model.device)  # Shape: [batch_size]
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=self.model.device)  # Shape: [batch_size]

                
                # Compute Q-values (estimates) for the current state using the policy network
                q_values = self.model(normalized_close_prices_tensor, account_balances_tensor, shares_held_tensor)  # Shape: [batch_size, num_stocks, num_actions_per_stock]
                # Shape: [batch_size, num_stocks, 1]
                
                actions_selected = batch_actions_tensor.unsqueeze(-1)
                
                
                # Select the Q-values for the actions taken
                # Add batch dimension
                
                # Adds an axis for gather, Shape: [batch_size, num_stocks, 1]
                estimates = q_values.gather(-1, actions_selected).squeeze(-1)  # Removes the last axis, Shape: [batch_size, num_stocks]

                # Compute the Q-values for the next state using the target network, and detach to stop gradients
                next_q_values = self.target_model(next_normalized_close_prices_tensor, next_account_balances_tensor, next_shares_held_tensor)  # Shape: [batch_size, num_stocks, num_actions_per_stock]

                # Select the maximum Q-value at the next state for each stock
                next_actions = next_q_values.max(dim=-1)[0]  # Shape: [batch_size, num_stocks]

                # Compute the target Q-values for each stock
                targets = batch_rewards_tensor.unsqueeze(-1) + gamma * (1 - batch_dones_tensor.unsqueeze(-1)) * next_actions  # Shape: [batch_size, num_stocks]

                # Flatten the tensors to use in loss computation
                estimates_flat = estimates.view(-1)  # Shape: [batch_size * num_stocks]
                targets_flat = targets.view(-1)  # Shape: [batch_size * num_stocks]


                loss = self.loss_fn(targets_flat, estimates_flat)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()


                if iteration % self.target_q_network_sync_period == 0:
                    with torch.no_grad():
                        for target_param, source_param in zip(self.target_model.parameters(), self.model.parameters()):
                            target_param.copy_(source_param)
                if done:
                    # print("It is done ")
                    break

                state = next_state
                iteration += 1
            percentage_of_random_choices = np.mean(np.array(self.random_choice_list[-1]))
            print(f"Episode {episode_index}, reward: {episode_reward}, percentage of random = {percentage_of_random_choices}")
            episode_reward_list.append(episode_reward)
            self.epsilon_greedy.decay_epsilon()
        return episode_reward_list
        
    def test(self, env):
        '''Test the agent on a multi-stock environment.'''
        state = env.reset()
        done = False
        episode_reward = 0.
        while not done:
            action = self.epsilon_greedy(state, test=True)  # Ensure test mode doesn't use epsilon
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        return episode_reward
    
   

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
    transformed_states = []

    # Iterate through each element in the batch
    for i in range(len(state_balances)):  # Assuming state_balances has one entry per batch element
        new_state_format = []
        normalised_close_prices = []
        for t in range(length_of_sequence):
            for s in range(d):
                # Append the price for stock s at time t
                normalised_close_prices.append(state_prices[i][s][t])
        
        # Add the account balance and shares held to the end of the state, as per original format
        new_state_format.append(normalised_close_prices)
        new_state_format.append(state_balances[i])  # Account balance
        new_state_format.append(state_shares_held[i])  # Shares held for each stock (if applicable)
        
        transformed_states.append(new_state_format)

    
    return transformed_states, length_of_sequence
