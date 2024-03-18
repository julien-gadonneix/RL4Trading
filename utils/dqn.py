import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):
    def __init__(self, input_size_price_list, input_size_balance_value, input_size_num_of_shares, device):
        super(DQN, self).__init__()
        self.device = device
        self.output_lstm_shape = 64
        self.hidden_layer_shape = 32
        self.output_shape = 3
        self.numlayers = 2
        
        self.lstm = nn.LSTM(input_size_price_list,  self.output_lstm_shape, self.numlayers, batch_first=True)
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

