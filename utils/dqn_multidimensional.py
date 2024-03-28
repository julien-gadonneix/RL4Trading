import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiStockDQN(nn.Module):
    def __init__(self, num_stocks, input_size_per_stock, input_size_balance_value, input_size_num_of_shares, device):
        super(MultiStockDQN, self).__init__()
        self.device = device
        self.num_stocks = num_stocks
        self.input_size_per_stock = input_size_per_stock
        self.output_lstm_shape = 64
        self.hidden_layer_shape = 32
        self.output_shape = 3 * num_stocks  # Assuming you want separate actions for each stock
        self.numlayers = 2
        
        # LSTM input feature size is now num_stocks * input_size_per_stock, as we concatenate features of all stocks at each timestep
        self.lstm = nn.LSTM(num_stocks * input_size_per_stock, self.output_lstm_shape, self.numlayers, batch_first=True)
        
        # Adjust the input size for the first fully connected layer accordingly
        self.fc1 = nn.Linear(self.output_lstm_shape + input_size_balance_value + input_size_num_of_shares, self.hidden_layer_shape)
        
        # Output layer - separate action for each stock
        self.fc2 = nn.Linear(self.hidden_layer_shape, self.output_shape)

    def forward(self, x_price_tensor, x_portfolio_value_tensor, x_num_of_shares_tensor):
        """
        Forward pass through the network.

        Args:
            x_price_tensor (torch.tensor): Tensor containing the concatenated stock prices over time for all stocks.
                                           Shape: (batch_size, sequence_length, num_stocks * input_size_per_stock)
            x_portfolio_value_tensor (torch.tensor): Tensor containing the portfolio value information.
                                                     Shape: (batch_size, 1)
            x_num_of_shares_tensor (torch.tensor): Tensor containing the number of shares information for each stock.
                                                   Shape: (batch_size, num_stocks)

        Returns:
            torch.tensor: Actions to be taken for each stock. Shape: (batch_size, num_stocks * 3)
        """
        # LSTM processing
        num_actions_per_stock = 3
        out, _ = self.lstm(x_price_tensor)
        x = out[:, -1, :]  # Taking the output of the last time step

        # Prepare additional inputs
        x_portfolio_value_tensor = x_portfolio_value_tensor.view(-1, 1)
        x_num_of_shares_tensor = x_num_of_shares_tensor.view(-1, self.num_stocks)  # Make sure this matches the expected shape

        # Concatenate all inputs
        x = torch.cat((x, x_portfolio_value_tensor, x_num_of_shares_tensor), 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)
        

        # Reshape the output so that it matches the expected shape (batch_size, num_stocks, num_actions_per_stock)
        x = x.view(-1, self.num_stocks, num_actions_per_stock)
        
        return x
