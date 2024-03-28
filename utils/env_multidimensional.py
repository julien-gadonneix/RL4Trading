import gym
import numpy as np
from gym import spaces

class MultiStockTradingEnv(gym.Env):
    '''A multi-stock trading environment for reinforcement learning.'''

    def __init__(self, dfs, start_balance=1000., T=20, d=10):
        super(MultiStockTradingEnv, self).__init__()

        self.dfs = dfs  # dfs is now a list of dataframes, each representing a stock
        self.start_balance = start_balance
        self.T = T
        self.d = d  # Number of different stocks
        
        # Find max close price across all stocks for normalization
        self.max_share_price = max(float(df['Close'].max().iloc[0]) for df in dfs)
        for df in dfs:
            df['Close normalized'] = df["Close"] / self.max_share_price
        
        # Actions: array of real numbers between -1 and 1 for each stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(d,), dtype=np.float32)

        # Observations: array that includes normalized close price for T previous days, 
        # account balance, and shares held for each stock
        self.observation_space = spaces.Box(low=0, high=1, shape=((d*2+1,)), dtype=np.float32)

        self.current_step = 0
        self.balance = start_balance
        self.shares_held = np.zeros(d)

   
    def _next_observation(self):
        """Returns the next observation for all stocks."""
        normalized_close_price_list= []
        for df in self.dfs:
            normalized_close_price = df.iloc[self.current_step-self.T:self.current_step+1]['Close normalized'].tolist()
            normalized_close_price_list.append(normalized_close_price)
        account_balance = self.balance
        shares_held = self.shares_held
        return normalized_close_price_list, account_balance, shares_held

    
    

    def _take_action(self, actions, props):
        """Updates the agent's balance and shares based on the actions for each stock,
        taking into account the expected GAIN of the actions. Sell actions are processed
        before buy actions."""
        # First, process all sell actions
        for i, action in enumerate(actions):
            if action == 0:  # Sell action
                current_price = float(self.dfs[i].iloc[self.current_step]['Close normalized'].iloc[0])
                shares_sold = -action * self.shares_held[i]  # Sell all shares
                self.balance += shares_sold * current_price
                self.shares_held[i] -= shares_sold

        # Calculate the total probability of buying actions to normalize buying amounts
        total_buying_prop = sum(prop for action, prop in zip(actions, props) if action == 2)
        balance_before_buy = self.balance
    
        # Then, process all buy actions with the updated balance
        for i, (action, prop) in enumerate(zip(actions, props)):
            if action == 2:  # Buy action
                current_price = float(self.dfs[i].iloc[self.current_step]['Close normalized'].iloc[0])
                if total_buying_prop > 0:  # Prevent division by zero
                    # Calculate the proportion of this action's expected GAIN out of the total buying expected GAIN
                    normalized_prob = prop / total_buying_prop
                    # Calculate the total possible investment based on the normalized expected GAIN and current balance
                    investment = balance_before_buy * normalized_prob
                    shares_bought = investment / current_price
                    # Update balance and shares held
                    self.balance -= shares_bought * current_price
                    self.shares_held[i] += shares_bought
    

    def step(self, actions,props):
        """Executes a step in the environment for all stocks."""
        previous_wealth = self.balance + sum(self.shares_held[i] * float(self.dfs[i].iloc[self.current_step]['Close normalized'].iloc[0]) for i in range(self.d))
        
        self._take_action(actions,props)
        self.current_step += 1

        if self.current_step > len(self.dfs[0]) - 2:  # Assuming all dfs have the same length
            done = True
        else:
            done = False

        current_wealth = self.balance + sum(self.shares_held[i] * float(self.dfs[i].iloc[self.current_step]['Close normalized'].iloc[0])for i in range(self.d))
        reward = current_wealth - previous_wealth

        if float(current_wealth) < 1e-12:
            done = True
        obs = self._next_observation()
        return obs, reward, done

    def reset(self):
        """Resets the environment to its initial state for all stocks."""
        self.balance = self.start_balance
        self.shares_held = np.zeros(self.d)
        self.current_step = self.T
        return self._next_observation()
