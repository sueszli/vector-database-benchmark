"""SVD of linear degradation matrices described in the paper
    ``Denoising Diffusion Restoration Models.''
    @article{kawar2022denoising,
      title={Denoising Diffusion Restoration Models},
      author={Bahjat Kawar and Michael Elad and Stefano Ermon and Jiaming Song},
      year={2022},
      journal={arXiv preprint arXiv:2201.11793},
    }
"""
import torch
__all__ = ['SVD', 'IdentitySVD', 'DenoiseSVD', 'ColorizationSVD']

class SVD(object):
    """SVD decomposition of a matrix, i.e., H = UDV^T.
        NOTE: assume that all inputs (i.e., h, x) are of shape [B, CHW].
    """

    def __init__(self, h):
        if False:
            while True:
                i = 10
        (self.u, self.d, self.v) = torch.svd(h, some=False)
        self.ut = self.u.t()
        self.vt = self.v.t()
        self.d[self.d < 0.001] = 0

    def U(self, x):
        if False:
            return 10
        return torch.matmul(self.u, x)

    def Ut(self, x):
        if False:
            i = 10
            return i + 15
        return torch.matmul(self.ut, x)

    def V(self, x):
        if False:
            print('Hello World!')
        return torch.matmul(self.v, x)

    def Vt(self, x):
        if False:
            return 10
        return torch.matmul(self.vt, x)

    @property
    def D(self):
        if False:
            return 10
        return self.d

    def H(self, x):
        if False:
            return 10
        return self.U(self.D * self.Vt(x)[:, :self.D.size(0)])

    def Ht(self, x):
        if False:
            print('Hello World!')
        return self.V(self._pad(self.D * self.Ut(x)[:, :self.D.size(0)]))

    def Hinv(self, x):
        if False:
            print('Hello World!')
        'Multiplies x by the pseudo inverse of H.\n        '
        x = self.Ut(x)
        x[:, :self.D.size(0)] = x[:, :self.D.size(0)] / self.D
        return self.V(self._pad(x))

    def _pad(self, x):
        if False:
            while True:
                i = 10
        o = x.new_zeros(x.size(0), self.v.size(0))
        o[:, :self.u.size(0)] = x.view(x.size(0), -1)
        return o

    def to(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Update the data type and device of UDV matrices.\n        '
        for (k, v) in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(*args, **kwargs))
        return self

class IdentitySVD(SVD):

    def __init__(self, c, h, w):
        if False:
            while True:
                i = 10
        self.d = torch.ones(c * h * w)

    def U(self, x):
        if False:
            return 10
        return x.clone()

    def Ut(self, x):
        if False:
            return 10
        return x.clone()

    def V(self, x):
        if False:
            return 10
        return x.clone()

    def Vt(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x.clone()

    def H(self, x):
        if False:
            while True:
                i = 10
        return x.clone()

    def Ht(self, x):
        if False:
            i = 10
            return i + 15
        return x.clone()

    def Hinv(self, x):
        if False:
            i = 10
            return i + 15
        return x.clone()

    def _pad(self, x):
        if False:
            i = 10
            return i + 15
        return x.clone()

class DenoiseSVD(SVD):

    def __init__(self, c, h, w):
        if False:
            for i in range(10):
                print('nop')
        self.num_entries = c * h * w
        self.d = torch.ones(self.num_entries)

    def U(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x.clone()

    def Ut(self, x):
        if False:
            while True:
                i = 10
        return x.clone()

    def V(self, x):
        if False:
            i = 10
            return i + 15
        return x.clone()

    def Vt(self, x):
        if False:
            while True:
                i = 10
        return x.clone()

    def _pad(self, x):
        if False:
            i = 10
            return i + 15
        return x.clone()

class ColorizationSVD(SVD):

    def __init__(self, c, h, w):
        if False:
            return 10
        self.color_dim = c
        self.num_pixels = h * w
        (self.u, self.d, self.v) = torch.svd(torch.ones(1, c) / c, some=False)
        self.vt = self.v.t()

    def U(self, x):
        if False:
            return 10
        return self.u[0, 0] * x

    def Ut(self, x):
        if False:
            i = 10
            return i + 15
        return self.u[0, 0] * x

    def V(self, x):
        if False:
            while True:
                i = 10
        return torch.einsum('ij,bjn->bin', self.v, x.view(x.size(0), self.color_dim, self.num_pixels)).flatten(1)

    def Vt(self, x):
        if False:
            i = 10
            return i + 15
        return torch.einsum('ij,bjn->bin', self.vt, x.view(x.size(0), self.color_dim, self.num_pixels)).flatten(1)

    @property
    def D(self):
        if False:
            return 10
        return self.d.repeat(self.num_pixels)

    def _pad(self, x):
        if False:
            print('Hello World!')
        o = x.new_zeros(x.size(0), self.color_dim * self.num_pixels)
        o[:, :self.num_pixels] = x
        return o