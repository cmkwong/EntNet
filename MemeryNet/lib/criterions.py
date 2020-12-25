import torch
import torch.nn as nn

class Loss_Calculator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, ans):
        """
        :param predict: torch.tensor(size=(n,1))
        :param ans: torch.tensor(size=(n,1))
        :return: torch.tensor(scalar)
        """
        loss = torch.mm(predict.t(), ans).sigmoid().log().squeeze(dim=0)
        return loss
