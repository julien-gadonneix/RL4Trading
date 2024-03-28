import numpy as np
import torch
from tqdm.notebook import tqdm
import itertools

SAVE_MODEL_EVERY = 50


class Agent:
    '''Agent class to interact with the environment.
    Args:
        env (Environment): Environment to interact with.
        model (nn.Module): Model to use for the agent.
        target_model (nn.Module): Target model to use for the agent.
        target_q_network_sync_period (int): Period to sync the target model with the model.
        optimizer (Optimizer): Optimizer to use for the agent.
        lr_scheduler (LRScheduler): Learning rate scheduler to use for the agent.
        loss_fn (function): Loss function to use for the agent.
        replay_buffer (ReplayBuffer): Replay buffer to use for the agent.
        epsilon_greedy (EpsilonGreedy): Epsilon greedy strategy to use for the agent.

    Attributes:
        env (Environment): Environment to interact with.
        model (nn.Module): Model to use for the agent.
        target_model (nn.Module): Target model to use for the agent.
        target_q_network_sync_period (int): Period to sync the target model with the model.
        optimizer (Optimizer): Optimizer to use for the agent.
        lr_scheduler (LRScheduler): Learning rate scheduler to use for the agent.
        loss_fn (function): Loss function to use for the agent.
        replay_buffer (ReplayBuffer): Replay buffer to use for the agent.
        epsilon_greedy (EpsilonGreedy): Epsilon greedy strategy to use for the agent.
        action_list (list): List of actions taken in each episode.
        wealth_list (list): List of wealths obtained in each episode.
        random_choice_list (list): List of random choices made in each episode.

    Methods:
        train: Trains the agent.
        test: Tests the agent.
    
    '''

    def __init__(self, env, model, type_of_model, target_model, target_q_network_sync_period, optimizer, lr_scheduler, loss_fn, replay_buffer, epsilon_greedy):
        self.env = env
        self.model = model
        self.type_of_model = type_of_model
        self.target_model = target_model
        self.target_q_network_sync_period = target_q_network_sync_period
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.replay_buffer = replay_buffer
        self.epsilon_greedy = epsilon_greedy
        self.action_list = list()
        self.wealth_list = list()
        self.random_choice_list = list()
        

    def train(self, num_episodes, gamma, batch_size):
        '''Trains the agent
        Args:
            num_episodes (int): Number of episodes to train the agent for.
            gamma (float): Discount factor.
            batch_size (int): Batch size for training the agent.

        Returns:
            episode_reward_list (list): List of rewards obtained in each episode.
        '''
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

                action, prop, is_random_choice = self.epsilon_greedy(state, self.type_of_model)
                self.random_choice_list[-1].append(is_random_choice)
                self.action_list[-1].append(action)
                next_state, reward, done = self.env.step(action, prop)
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
                self.wealth_list[-1].append(1000+episode_reward)
 

                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(min(batch_size, len(self.replay_buffer)))
                
                
                normalized_close_prices , account_balances, shares_held = zip(*batch_states)
                normalized_close_prices = np.array(normalized_close_prices)
                # reshape coordinate number 1
                normalized_close_prices = normalized_close_prices.reshape(normalized_close_prices.shape[0], normalized_close_prices.shape[1], 1)
                account_balances = np.array(account_balances)
                shares_held = np.array(shares_held)

                normalized_close_prices_tensor = torch.tensor(normalized_close_prices, dtype=torch.float32, device=self.model.device)
                # reshape from (21,1,128) to (128,1,21)

                account_balances_tensor = torch.tensor(account_balances, dtype=torch.float32, device=self.model.device)
                shares_held_tensor = torch.tensor(shares_held, dtype=torch.float32, device=self.model.device)

                next_normalized_close_prices , next_account_balances, next_shares_held = zip(*batch_next_states)
                next_normalized_close_prices = np.array(next_normalized_close_prices)
                next_normalized_close_prices = next_normalized_close_prices.reshape(next_normalized_close_prices.shape[0], next_normalized_close_prices.shape[1], 1)
                next_account_balances = np.array(next_account_balances)
                next_shares_held = np.array(next_shares_held)

                next_normalized_close_prices_tensor = torch.tensor(next_normalized_close_prices, dtype=torch.float32, device=self.model.device)
                # reshape from (21,1,128) to (128,1,21)
                next_account_balances_tensor = torch.tensor(next_account_balances, dtype=torch.float32, device=self.model.device)
                next_shares_held_tensor = torch.tensor(next_shares_held, dtype=torch.float32, device=self.model.device)


                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.model.device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.model.device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=self.model.device)

                sequence_length = normalized_close_prices_tensor.size(1)

                if self.type_of_model == "DQN" :

                    estimates = self.model(normalized_close_prices_tensor,
                                            account_balances_tensor, 
                                            shares_held_tensor
                                            ).gather(1, batch_actions_tensor.unsqueeze(1))
                    next_actions = self.target_model(next_normalized_close_prices_tensor,
                                                    next_account_balances_tensor, 
                                                    next_shares_held_tensor
                                                    ).max(dim=1)[0]

                elif self.type_of_model == 'DQN_with_Transformer':

                    tgt_mask = self.model.get_tgt_mask(sequence_length).to(self.model.device)

                    estimates = self.model(normalized_close_prices_tensor,
                                            account_balances_tensor, 
                                            shares_held_tensor,
                                            tgt_mask
                                            ).gather(1, batch_actions_tensor.unsqueeze(1))
                    next_actions = self.target_model(next_normalized_close_prices_tensor,
                                                    next_account_balances_tensor, 
                                                    next_shares_held_tensor,
                                                    tgt_mask
                                                    ).max(dim=1)[0]
                targets = batch_rewards_tensor + gamma*(1-batch_dones_tensor)*next_actions
                targets = targets.unsqueeze(1)
                loss = self.loss_fn(targets, estimates)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                


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
            print(f"Ep {episode_index}, reward: {episode_reward:.2f}, % of random = {percentage_of_random_choices:.2f}, lr = {self.optimizer.param_groups[0]['lr']:.2e}, % sell= {np.mean(np.array(self.action_list[-1])==0):.2f}, % hold= {np.mean(np.array(self.action_list[-1])==1):.2f}, % buy= {np.mean(np.array(self.action_list[-1])==2):.2f}")
            episode_reward_list.append(episode_reward)
            self.epsilon_greedy.decay_epsilon()
            self.lr_scheduler.step()
            # save model every 20 episodes
            if episode_index % SAVE_MODEL_EVERY == 0:
                print(f"Saving model at episode {episode_index}")
                torch.save(self.model.state_dict(), f"model_{episode_index}.pt")

        return episode_reward_list
    

    def test(self, env):
        '''Tests the agent.
        Args:
            env (Environment): Environment to test the agent on.
            
        Returns:
            reward_list (list): List of rewards obtained in the episode.
            action_list (list): List of actions taken in the episode.
            prop_list (list): List of propensities of the actions taken in the episode.
        '''
        state = env.reset()
        reward_list = []
        action_list = []
        prop_list = []
        done = False
        self.epsilon_greedy.epsilon = 0.
        while not done:
            action, prop, is_random_choice = self.epsilon_greedy(state, self.type_of_model)
            next_state, reward, done = env.step(action, prop)
            action_list.append(action)
            prop_list.append(prop)
            reward_list.append(reward)
            print(f"Action: {action}, Reward: {reward:.2f}, Random Choice: {is_random_choice}, Done: {done}")
            state = next_state
        return reward_list, action_list, prop_list



