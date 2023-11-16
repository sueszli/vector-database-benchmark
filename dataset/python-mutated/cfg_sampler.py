from copy import deepcopy
import torch.nn as nn

class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        if False:
            print('Hello World!')
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + y['scale'].view(-1, 1, 1, 1) * (out - out_uncond)