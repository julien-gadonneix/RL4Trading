import gym
import numpy as np
from gym import spaces

 
class SimplifiedStockTradingEnv(gym.Env):
    '''A simplified stock trading environment for reinforcement learning.'''


    def __init__(self, df, start_balance=1000., T=20, d=1):
        super(SimplifiedStockTradingEnv, self).__init__()

        self.df = df
        self.start_balance = start_balance
        # Time range to look back
        self.T = T
        # Number of different stocks to look at
        self.d = d
        self.max_share_price = df['Close'].max()

        self.df['Close normalized']=self.df["Close"]/self.max_share_price
        
        # Actions: real number between -1 and 1
        # 0: Hold
        # Positive: Buy
        # 1: Buy as many shares as possible
        # Negative: Sell
        # -1: Sell all possessed shares
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observations: arrray of T+2 elements
        # Normalised close price of the T previous days (sliding window)
        # Account balance
        # Number of shares held
        self.observation_space = spaces.Box(low=0, high=1, shape=(T+2,), dtype=np.float32)

        self.current_step = 0
        self.balance = start_balance
        self.shares_held = 0.


    def _next_observation(self):
        """Returns the next observation."""
        normalized_close_price = self.df.iloc[self.current_step-self.T:self.current_step+1]['Close normalized'].tolist() 
        account_balance = self.balance
        shares_held = self.shares_held 
        return normalized_close_price, account_balance, shares_held 
    

    def _take_action(self, action):
        """Updates the agent's balance and shares based on the action."""
        current_price = self.df.iloc[self.current_step]['Close normalized']

        if action > 0:  # Buy
            total_possible = self.balance / current_price
            shares_bought = action * total_possible
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought

        elif action < 0:  # Sell
            shares_sold = -action * self.shares_held
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold


    def step(self, action):
        """Executes a step in the environment."""
        previous_wealth = self.balance + self.shares_held * self.df.iloc[self.current_step]['Close normalized']
        
        if self.current_step > len(self.df) - 2 : # end of data
            # print("End of data")
            done = True
            current_wealth = previous_wealth
            reward = 0.
            obs = self._next_observation()
        else : 
            self._take_action(action)
            obs = self._next_observation()
            self.current_step += 1
            current_wealth = self.balance + self.shares_held * self.df.iloc[self.current_step]['Close normalized']
            reward = current_wealth - previous_wealth
            if current_wealth<1e-12 : 
                done = True
            else :
                done = False

        return obs, reward, done


    def reset(self):
        """Resets the environment to its initial state."""
        self.balance = self.start_balance
        self.shares_held = 0.
        self.current_step = self.T
        return self._next_observation()
