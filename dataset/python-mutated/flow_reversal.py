import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowReversal(nn.Module):
    """docstring for WarpLayer"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(FlowReversal, self).__init__()

    def forward(self, img, flo):
        if False:
            i = 10
            return i + 15
        '\n            -img: image (N, C, H, W)\n            -flo: optical flow (N, 2, H, W)\n            elements of flo is in [0, H] and [0, W] for dx, dy\n\n        '
        (N, C, _, _) = img.size()
        y = flo[:, 0:1, :]
        x = flo[:, 1:2, :, :]
        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1
        (w11, w12, w21, w22) = self.get_gaussian_weights(x, y, x1, x2, y1, y2)
        (img11, o11) = self.sample_one(img, x1, y1, w11)
        (img12, o12) = self.sample_one(img, x1, y2, w12)
        (img21, o21) = self.sample_one(img, x2, y1, w21)
        (img22, o22) = self.sample_one(img, x2, y2, w22)
        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22
        return (imgw, o)

    def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
        if False:
            while True:
                i = 10
        w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
        w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
        w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
        w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))
        return (w11, w12, w21, w22)

    def sample_one(self, img, shiftx, shifty, weight):
        if False:
            while True:
                i = 10
        '\n        Input:\n            -img (N, C, H, W)\n            -shiftx, shifty (N, c, H, W)\n        '
        (N, C, H, W) = img.size()
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].cuda().long().repeat(N, C, 1, W).view(-1)
        flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].cuda().long().repeat(N, C, H, 1).view(-1)
        flat_weight = weight.view(-1)
        flat_img = img.view(-1)
        idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().cuda().repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().cuda().repeat(N, 1, H, W).view(-1)
        idxx = flat_shiftx.long() + flat_basex
        idxy = flat_shifty.long() + flat_basey
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)
        ids = idxn * C * H * W + idxc * H * W + idxx * W + idxy
        ids_mask = torch.masked_select(ids, mask).clone().cuda()
        img_warp = torch.zeros([N * C * H * W]).cuda()
        img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)
        one_warp = torch.zeros([N * C * H * W]).cuda()
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)
        return (img_warp.view(N, C, H, W), one_warp.view(N, C, H, W))