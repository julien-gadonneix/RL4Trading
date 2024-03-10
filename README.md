
# RL4Trading

In this project, we are trying to implement a Reinforcement Learning algorithm to financial data.

## Structure of the data

We are fetching data from yahoo finance, through the API, and we are looking to close prices in our model.

## Structure of the model

### Environement

Our environement is composed of :

* A dataframe with stock price data,
* A range of time which will be taken into account in the action process
* A balance for the money that cna be invested
* A number of shares held at every time steps

### Actions

Our action space is composed of 3 different types :

* Buy a certain proportion of stocks according to the confidence of the model to choose a buying strategy
* Sell a certain propoértion of stocks according to the confidence of the model to choose a selling strategy
* Do nothing

### Deep Q-Network

Our DQN is based on a mixture of layers especially LSTM because we are dealing with sequence of trading data, and other informations such as the number of stocks held, the account balance, aso.
