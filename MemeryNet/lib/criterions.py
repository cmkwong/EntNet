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
        # cosine loss
        predict_magnitude = predict.pow(2).sum(dim=0).sqrt()
        ans_magnitude = ans.pow(2).sum(dim=0).sqrt()
        unit_predict = predict / predict_magnitude
        unit_ans = ans / ans_magnitude
        cos_loss = -1 * torch.mm(unit_predict.t(), unit_ans).sigmoid().log().squeeze(dim=0)

        # magnitude loss
        # magnitude_loss = (predict_magnitude - ans_magnitude).abs()
        return  cos_loss
