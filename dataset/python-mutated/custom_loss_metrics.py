import torch
from torch.nn.modules.loss import _Loss

class PinballLoss(_Loss):
    """Class for the PinBall loss for quantile regression"""

    def __init__(self, loss_func, quantiles):
        if False:
            print('Hello World!')
        '\n        Args:\n            loss_func : torch.nn._Loss\n                Loss function to be used as the\n                base loss for pinball loss\n            quantiles : list\n                list of quantiles estimated from the model\n        '
        super().__init__()
        self.loss_func = loss_func
        self.quantiles = quantiles

    def forward(self, outputs, target):
        if False:
            print('Hello World!')
        '\n        Computes the pinball loss from forecasts\n        Args:\n            outputs : torch.tensor\n                outputs from the model of dims (batch, no_quantiles, n_forecasts)\n            target : torch.tensor\n                actual targets of dims (batch, n_forecasts)\n\n        Returns:\n            float\n                pinball loss\n        '
        target = target.repeat(1, 1, len(self.quantiles))
        differences = target - outputs
        base_losses = self.loss_func(outputs, target)
        positive_losses = torch.tensor(self.quantiles, device=target.device).unsqueeze(dim=0).unsqueeze(dim=0) * base_losses
        negative_losses = (1 - torch.tensor(self.quantiles, device=target.device).unsqueeze(dim=0).unsqueeze(dim=0)) * base_losses
        pinball_losses = torch.where(differences >= 0, positive_losses, negative_losses)
        multiplier = torch.ones(size=(1, 1, len(self.quantiles)), device=target.device)
        multiplier[:, :, 0] = 2
        pinball_losses = multiplier * pinball_losses
        return pinball_losses