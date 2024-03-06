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

        for episode_index in tqdm(range(1, num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.model.device)


            for t in itertools.count():
                action = self.epsilon_greedy(state_tensor)
                next_state, reward, done = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                episode_reward += reward

                if len(self.replay_buffer) > batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(batch_size)

                    batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=self.model.device)
                    batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.model.device)
                    batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.model.device)
                    batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=self.model.device)
                    batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=self.model.device)

                    estimates = self.model(batch_states_tensor)
                    next_actions = self.target_model(batch_next_states_tensor)
                    targets = batch_rewards_tensor + gamma*(1-batch_dones_tensor)*next_actions
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
                break

            state = next_state
            iteration += 1

        episode_reward_list.append(episode_reward)
        self.epsilon_greedy.decay_epsilon()
        return episode_reward_list


