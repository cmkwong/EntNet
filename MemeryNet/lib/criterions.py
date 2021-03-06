import torch
import torch.nn as nn

class CosineLoss(nn.Module):
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

class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nllloss = nn.NLLLoss()

    def forward(self, predict, ans):
        """
        :param predict: torch.tensor, shape=(1,m)
        :param ans: torch.tensor, shape=(1), value is from 0 to m-1
        :return: torch.tensor (scalar value)
        """
        loss = self.nllloss(predict, ans)
        return loss

class CrossEntropy_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossloss = nn.CrossEntropyLoss()

    def forward(self, predict, ans):
        """
        :param predict: torch.tensor, shape=(1,m)
        :param ans: torch.tensor, shape=(1), value is from 0 to m-1
        :return: torch.tensor (scalar value)
        """
        loss = self.crossloss(predict, ans)
        return loss