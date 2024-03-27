import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, number_of_data, input_size_price_list, input_size_balance_value, input_size_num_of_shares, device):
        super(DQN, self).__init__()
        self.device = device
        self.output_lstm_shape = 64
        self.hidden_layer_shape = 32
        self.output_shape = 3
        self.numlayers = 2
        
        self.lstm = nn.LSTM(number_of_data,  self.output_lstm_shape, self.numlayers, batch_first=True)
        self.fc1 = nn.Linear(self.output_lstm_shape + input_size_balance_value + input_size_num_of_shares, self.hidden_layer_shape)
        self.fc2 = nn.Linear(self.hidden_layer_shape, self.output_shape)

    def forward(self, x_price_tensor, x_portfolio_value_tensor, x_num_of_shares_tensor):
        """_summary_

        Args:
            x_price_tensor (torch.tensor): list of T last stock prices
            x_portfolio_value_tensor (torch.tensor): value of the balance at time t
            x_num_of_shares_tensor (torch.tensor): value of the number of share at time t

        Returns:
            torch.tensor: the action to be taken between -1 and 1 
        """
        # h0 = torch.zeros(self.numlayers, x_price_tensor.size(0), self.output_lstm_shape).to(self.device)
        # c0 = torch.zeros(self.numlayers, x_price_tensor.size(0), self.output_lstm_shape).to(self.device)
        out, _ = self.lstm(x_price_tensor) #, (h0, c0))
        x = out[:, -1, :]
        # add the portfolio value and the number of shares to the input
        x_portfolio_value_tensor = x_portfolio_value_tensor.view(-1, 1)
        x_num_of_shares_tensor = x_num_of_shares_tensor.view(-1, 1, )
        x = torch.cat((x, x_portfolio_value_tensor, x_num_of_shares_tensor), 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class DQN_with_Transformer(nn.Module):
    def __init__(self, number_of_data, input_size_price_list, input_size_balance_value, input_size_num_of_shares, device):
        super(DQN_with_Transformer, self).__init__()
        self.device = device
        self.hidden_layer_shape = 32
        self.output_shape = 3
        # Transformer layer expects input size in the format [sequence length, batch size, features]
        self.transformer = nn.Transformer(nhead=1, batch_first=True, d_model = 1, dropout=0.1)
        self.fc1 = nn.Linear(input_size_price_list + input_size_balance_value + input_size_num_of_shares, self.hidden_layer_shape)
        self.fc2 = nn.Linear(self.hidden_layer_shape, self.output_shape)

    def forward(self, x_price_tensor, x_portfolio_value_tensor, x_num_of_shares_tensor):
        """_summary_

        Args:
            x_price_tensor (torch.tensor): list of T last stock prices
            x_portfolio_value_tensor (torch.tensor): value of the balance at time t
            x_num_of_shares_tensor (torch.tensor): value of the number of share at time t

        Returns:
            torch.tensor: the action to be taken between -1 and 1 
        """

        # Apply the transformer layer
        x = self.transformer(x_price_tensor, x_price_tensor)
        # x is shape [1, 21, 1] and i want to reshape it to [1, 21]
        x = x.view(-1, 21) 
        # Add the portfolio value and the number of shares to the input
        x_portfolio_value_tensor = x_portfolio_value_tensor.view(-1, 1)
        x_num_of_shares_tensor = x_num_of_shares_tensor.view(-1, 1)

        # Concatenate inputs
        x = torch.cat((x, x_portfolio_value_tensor, x_num_of_shares_tensor), dim=1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # between -1 and 1
        return x