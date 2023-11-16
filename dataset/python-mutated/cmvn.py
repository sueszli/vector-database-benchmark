import re
import numpy as np
import torch

class GlobalCMVN(torch.nn.Module):

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool=True):
        if False:
            return 10
        '\n        Args:\n            mean (torch.Tensor): mean stats\n            istd (torch.Tensor): inverse std, std which is 1.0 / std\n        '
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.register_buffer('mean', mean)
        self.register_buffer('istd', istd)

    def forward(self, x: torch.Tensor):
        if False:
            return 10
        '\n        Args:\n            x (torch.Tensor): (batch, max_len, feat_dim)\n\n        Returns:\n            (torch.Tensor): normalized feature\n        '
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x

def load_kaldi_cmvn(cmvn_file):
    if False:
        print('Hello World!')
    ' Load the kaldi format cmvn stats file and no need to calculate\n\n    Args:\n        cmvn_file: cmvn stats file in kaldi format\n\n    Returns:\n        a numpy array of [means, vars]\n    '
    means = None
    variance = None
    with open(cmvn_file) as f:
        all_lines = f.readlines()
        for (idx, line) in enumerate(all_lines):
            if line.find('AddShift') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                means_str = re.findall('[\\[](.*?)[\\]]', next_line)[0]
                means_list = means_str.strip().split(' ')
                means = [0 - float(s) for s in means_list]
                assert len(means) == int(segs[1])
            elif line.find('Rescale') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                vars_str = re.findall('[\\[](.*?)[\\]]', next_line)[0]
                vars_list = vars_str.strip().split(' ')
                variance = [float(s) for s in vars_list]
                assert len(variance) == int(segs[1])
            elif line.find('Splice') != -1:
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                splice_str = re.findall('[\\[](.*?)[\\]]', next_line)[0]
                splice_list = splice_str.strip().split(' ')
                assert len(splice_list) * int(segs[2]) == int(segs[1])
                copy_times = len(splice_list)
            else:
                continue
    cmvn = np.array([means, variance])
    cmvn = np.tile(cmvn, (1, copy_times))
    return cmvn