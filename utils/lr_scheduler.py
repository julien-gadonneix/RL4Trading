import torch
from typing import List


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    '''A learning rate scheduler that does not go below a minimum learning rate.'''
    
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        '''Initialize a new instance of MinimumExponentialLR.'''
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)


    def get_lr(self) -> List[float]:
        '''Compute learning rate using chainable form of the scheduler.'''
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]