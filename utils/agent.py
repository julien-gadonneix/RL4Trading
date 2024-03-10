import numpy as np
import torch
from tqdm.notebook import tqdm
import itertools


class Agent:
    '''Agent class to interact with the environment.'''

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
        

    def train(self, num_episodes, gamma, batch_size):
        '''Trains the agent.'''
        iteration = 0
        episode_reward_list = []
        episode_reward = 0.

        for episode_index in tqdm(range(1, num_episodes+1)):
            # print(f"Episode {episode_index}")
            state = self.env.reset()
            episode_reward = 0.
            for iteration in itertools.count():

                action, prop = self.epsilon_greedy(state)
                next_state, reward, done = self.env.step(action, prop)
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward
 

                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(min(batch_size, len(self.replay_buffer)))
                
                
                normalized_close_prices , account_balances, shares_held = zip(*batch_states)
                normalized_close_prices = np.array(normalized_close_prices)
                # reshape coordinate number 1
                normalized_close_prices = normalized_close_prices.reshape(normalized_close_prices.shape[0], 1, normalized_close_prices.shape[1])
                account_balances = np.array(account_balances)
                shares_held = np.array(shares_held)

                normalized_close_prices_tensor = torch.tensor(normalized_close_prices, dtype=torch.float32, device=self.model.device)
                # reshape from (21,1,128) to (128,1,21)

                account_balances_tensor = torch.tensor(account_balances, dtype=torch.float32, device=self.model.device)
                shares_held_tensor = torch.tensor(shares_held, dtype=torch.float32, device=self.model.device)

                next_normalized_close_prices , next_account_balances, next_shares_held = zip(*batch_next_states)
                next_normalized_close_prices = np.array(next_normalized_close_prices)
                next_normalized_close_prices = next_normalized_close_prices.reshape(next_normalized_close_prices.shape[0], 1, next_normalized_close_prices.shape[1])
                next_account_balances = np.array(next_account_balances)
                next_shares_held = np.array(next_shares_held)

                next_normalized_close_prices_tensor = torch.tensor(next_normalized_close_prices, dtype=torch.float32, device=self.model.device)
                # reshape from (21,1,128) to (128,1,21)
                next_account_balances_tensor = torch.tensor(next_account_balances, dtype=torch.float32, device=self.model.device)
                next_shares_held_tensor = torch.tensor(next_shares_held, dtype=torch.float32, device=self.model.device)


                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.model.device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.model.device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=self.model.device)

                estimates = self.model(normalized_close_prices_tensor,
                                        account_balances_tensor, 
                                        shares_held_tensor
                                        ).gather(1, batch_actions_tensor.unsqueeze(1))
                next_actions = self.target_model(next_normalized_close_prices_tensor,
                                                next_account_balances_tensor, 
                                                next_shares_held_tensor
                                                ).max(dim=1)[0]
                targets = batch_rewards_tensor + gamma*(1-batch_dones_tensor)*next_actions
                targets = targets.unsqueeze(1)
                loss = self.loss_fn(targets, estimates)

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
            print(f"Episode reward: {episode_reward}")
            episode_reward_list.append(episode_reward)
            self.epsilon_greedy.decay_epsilon()
        return episode_reward_list


