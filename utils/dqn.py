import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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
        x = self.fc2(x)
        return x

class DQN_with_Transformer(nn.Module):
    def __init__(self, number_of_data, input_size_price_list, input_size_balance_value, input_size_num_of_shares, device):
        super(DQN_with_Transformer, self).__init__()
        self.device = device
        self.hidden_layer_shape = 32
        self.output_shape = 3
        self.dim_model = number_of_data
        self.lengh_of_sequence = input_size_price_list
        # Transformer layer expects input size in the format [sequence length, batch size, features]
        self.positional_encoder = PositionalEncoding(
            dim_model=self.dim_model, dropout_p=0.1, max_len=5000
        )
        self.transformer = nn.Transformer(nhead=1, batch_first=True, d_model = self.dim_model, dropout=0.1)
        self.fc1 = nn.Linear(input_size_price_list + input_size_balance_value + input_size_num_of_shares, self.hidden_layer_shape)
        self.fc2 = nn.Linear(self.hidden_layer_shape, self.output_shape)

    def forward(self, x_price_tensor, x_portfolio_value_tensor, x_num_of_shares_tensor, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """_summary_

        Args:
            x_price_tensor (torch.tensor): list of T last stock prices
            x_portfolio_value_tensor (torch.tensor): value of the balance at time t
            x_num_of_shares_tensor (torch.tensor): value of the number of share at time t

        Returns:
            torch.tensor: the action to be taken between -1 and 1 
        """
        
        # Apply the positional encoding
        x_price_tensor = self.positional_encoder(x_price_tensor)
        # Apply the transformer layer
        x = self.transformer(x_price_tensor, x_price_tensor, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # x is shape [1, 21, 1] and i want to reshape it to [1, 21]
        x = x.view(-1, self.lengh_of_sequence) 
        # Add the portfolio value and the number of shares to the input
        x_portfolio_value_tensor = x_portfolio_value_tensor.view(-1, 1)
        x_num_of_shares_tensor = x_num_of_shares_tensor.view(-1, 1)

        # Concatenate inputs
        x = torch.cat((x, x_portfolio_value_tensor, x_num_of_shares_tensor), dim=1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])