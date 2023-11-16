import math
import torch

def softmax_w_top(x, top):
    if False:
        while True:
            i = 10
    (values, indices) = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp)
    return x

def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    if False:
        while True:
            i = 10
    (yv, xv) = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    yv = yv.reshape(height * width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height * width).unsqueeze(0).float().cuda()
    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)
    g = torch.exp(-((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
    return g

def kmn(x, top=None, gauss=None):
    if False:
        for i in range(10):
            print('nop')
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes) * gauss
            (x_exp, indices) = torch.topk(x_exp, k=top, dim=1)
        else:
            (values, indices) = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:, 0])
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(1, indices, x_exp)
        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x - maxes) * gauss
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp
    return output

class MemoryBank:

    def __init__(self, compress, k, top_k=20, mode='stm'):
        if False:
            while True:
                i = 10
        self.top_k = top_k
        self.CK = None
        self.CV = None
        self.mem_k = None
        self.mem_v = None
        self.num_objects = k
        self.km = 5.6
        self.compress = compress
        self.init_mode(mode)

    def init_mode(self, mode):
        if False:
            while True:
                i = 10
        '\n        stm, two-frames, gt, last, compress, gt-compress,\n        last-compress, two-frames-compress\n        '
        self.is_compress = None
        self.use_gt = None
        self.use_last = None
        self.stm = None
        print('mode is {}'.format(mode))
        if mode == 'stm':
            self.stm = True
        elif mode == 'two-frames':
            self.use_gt = True
            self.use_last = True
        elif mode == 'gt':
            self.use_gt = True
        elif mode == 'last':
            self.use_last = True
        elif mode == 'compress':
            self.is_compress = True
        elif mode == 'gt-compress':
            self.use_gt = True
            self.is_compress = True
        elif mode == 'last-compress':
            self.use_last = True
            self.is_compress = True
        elif mode == 'two-frames-compress':
            self.use_gt = True
            self.use_last = True
            self.is_compress = True
        else:
            raise RuntimeError('check mode!')

    def _global_matching(self, mk, qk, H, W):
        if False:
            print('Hello World!')
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)
        (B, CK, NE) = mk.shape
        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        affinity = (-a + b) / math.sqrt(CK)
        affinity = softmax_w_top(affinity, top=self.top_k)
        return affinity

    def _readout(self, affinity, mv):
        if False:
            return 10
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        if False:
            while True:
                i = 10
        k = self.num_objects
        (_, _, h, w) = qk.shape
        qk = qk.flatten(start_dim=2)
        if self.temp_k is not None and self.is_compress and self.use_last and self.use_gt:
            mk = torch.cat([self.mem_k, self.temp_k, self.gt_k, self.gt_k], 2)
            try:
                mv = torch.cat([self.mem_v, self.temp_v, self.gt_v, self.gt_v], 2)
            except Exception:
                mv = torch.cat([self.mem_v, self.temp_v.unsqueeze(0), self.gt_v.unsqueeze(0), self.gt_v.unsqueeze(0)], 3)
        elif self.temp_k is not None and self.use_last and self.use_gt:
            mk = torch.cat([self.temp_k, self.gt_k, self.gt_k], 2)
            try:
                mv = torch.cat([self.temp_v, self.gt_v, self.gt_v], 2)
            except Exception:
                mv = torch.cat([self.temp_v.unsqueeze(0), self.gt_v.unsqueeze(0), self.gt_v.unsqueeze(0)], 3)
        elif self.temp_k is not None and self.is_compress and self.use_last:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            try:
                mv = torch.cat([self.mem_v, self.temp_v], 2)
            except Exception:
                mv = torch.cat([self.mem_v, self.temp_v.unsqueeze(0)], 3)
        elif self.is_compress and self.use_gt:
            mk = torch.cat([self.mem_k, self.gt_k], 2)
            try:
                mv = torch.cat([self.mem_v, self.gt_v], 2)
            except Exception:
                mv = torch.cat([self.mem_v, self.gt_v.unsqueeze(0)], 3)
        elif self.temp_k is not None and self.use_last:
            mk = self.temp_k
            mv = self.temp_v
        else:
            mk = self.mem_k
            mv = self.mem_v
        affinity = self._global_matching(mk, qk, h, w)
        if len(mv.shape) == 6:
            mv = mv.squeeze(0)
        mv = mv.flatten(start_dim=2)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)
        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        if False:
            i = 10
            return i + 15
        self.temp_k = None
        self.temp_v = None
        if self.mem_k is None:
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.gt_k = key
            self.gt_v = value
        elif self.is_compress:
            if len(self.mem_v.shape) == 5:
                self.mem_v = self.mem_v.unsqueeze(0)
            k = torch.cat([self.mem_k, key], 2)
            v = torch.cat([self.mem_v, value.unsqueeze(0)], 3)
            (self.mem_k, self.mem_v) = self.compress(k, v)
            if self.use_last:
                self.temp_k = key
                self.temp_v = value
        elif self.stm:
            self.mem_k = torch.cat([self.mem_k, key], 2)
            self.mem_v = torch.cat([self.mem_v, value], 2)
        elif self.use_last:
            self.temp_k = key
            self.temp_v = value