
# RL4Trading

In this project, we are implementing a Reinforcement Learning algorithm for financial data analysis.

## Data Structure

* We source our data from Yahoo Finance via their API, specifically focusing on the closing prices for our modeling purposes.

## Model Architecture

### Environment

* Our environment comprises:
  - A dataframe containing stock price data,
  - A specified time range for action consideration,
  - A balance representing investable funds,
  - A record of the number of shares held at each time step.

### Actions

* Our action space encompasses three distinct types:
  1. Buying a certain proportion of stocks based on the model's confidence in a buy strategy.
  2. Selling a certain proportion of stocks based on the model's confidence in a sell strategy.
  3. Taking no action.

### Deep Q-Network (DQN)

* Our DQN architecture is a blend of various layers, notably incorporating LSTM due to the sequential nature of trading data. Additionally, it considers other relevant factors such as the number of stocks held and the current account balance.
