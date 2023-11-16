from typing import Optional
import torch

def cov(x: torch.Tensor, rowvar: bool=False, bias: bool=False, ddof: Optional[int]=None, aweights: Optional[torch.Tensor]=None) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Estimates covariance matrix like ``numpy.cov``.\n    Arguments:\n        - x (:obj:`torch.Tensor`): A 1-D or 2-D tensor containing multiple variables and observations. Each row of             ``x`` represents a variable, and each column a single observation of all those variables.\n        - rowvar (:obj:`bool`): If ``rowvar`` is True by default, and each column is a single observation of all those             variables. Otherwise, each column represents a variable, while the rows contain observations.\n        - bias (:obj:`bool`): Default normalization (False) is by dividing ``N - 1``, where ``N`` is the number of             observations given (unbiased estimate). If ``bias`` is ``True``, then normalization is by ``N``.\n        - ddof (:obj:`Optional[int]`): If ``ddof`` is not ``None``, it implies that the argument ``bias`` is             overridden. Note that ``ddof=1`` will return the unbiased estimate (equals to ``bias=False``), and             ``ddof=0`` will return the biased estimation (equals to ``bias=True``).\n        - aweights (:obj:`Optional[torch.Tensor]`): 1-D tensor of observation vector weights. These relative weights             are typically large for observations considered “important” and smaller for observations considered less             “important”. If ``ddof=0``, the tensor of weights can be used to assign weights to observation vectors.\n    Returns:\n        - cov_mat (:obj:`torch.Tensor`): Covariance matrix calculated.\n    '
    if x.dim() == 1 and rowvar:
        raise NotImplementedError
    if x.dim() == 1:
        x = x.view(-1, 1)
    if rowvar and x.shape[0] != 1:
        x = x.t()
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum
    xm = x.sub(avg.expand_as(x))
    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()
    c = torch.mm(X_T, xm)
    c = c / fact
    return c.squeeze()