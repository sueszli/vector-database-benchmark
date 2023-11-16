import torch
import torch.nn.functional as F
from modelscope.models.cv.video_object_segmentation.aggregate import aggregate
from modelscope.models.cv.video_object_segmentation.inference_memory_bank import MemoryBank

def pad_divide_by(in_img, d, in_size=None):
    if False:
        i = 10
        return i + 15
    if in_size is None:
        (h, w) = in_img.shape[-2:]
    else:
        (h, w) = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    (lh, uh) = (int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2))
    (lw, uw) = (int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2))
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return (out, pad_array)

class InferenceCore:

    def __init__(self, prop_net, is_cuda, images, num_objects, top_k=20, mem_every=5, include_last=False):
        if False:
            print('Hello World!')
        self.prop_net = prop_net
        self.is_cuda = is_cuda
        self.mem_every = mem_every
        self.include_last = include_last
        t = images.shape[1]
        (h, w) = images.shape[-2:]
        (images, self.pad) = pad_divide_by(images, 16)
        (nh, nw) = images.shape[-2:]
        self.images = images
        if self.is_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.k = num_objects
        self.prob = torch.zeros((self.k + 1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-07
        (self.t, self.h, self.w) = (t, h, w)
        (self.nh, self.nw) = (nh, nw)
        self.kh = self.nh // 16
        self.kw = self.nw // 16
        self.mem_bank = MemoryBank(prop_net.mem_compress, k=self.k, top_k=top_k, mode='two-frames-compress')

    def encode_key(self, idx):
        if False:
            for i in range(10):
                print('nop')
        result = self.prop_net.encode_key(self.images[:, idx])
        return result

    def do_pass(self, first_k, first_v, idx, end_idx):
        if False:
            return 10
        global tt1, tt2, tt3, tt4
        self.mem_bank.add_memory(first_k, first_v)
        closest_ti = end_idx
        this_range = range(idx + 1, closest_ti)
        end = closest_ti - 1
        for ti in this_range:
            (k16, qv16, qf16, qf8, qf4) = self.encode_key(ti)
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16)
            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:, ti] = out_mask
            if ti != end:
                is_mem_frame = ti % self.mem_every == 0
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:, ti], qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, is_temp=not is_mem_frame)
        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        if False:
            return 10
        (mask, _) = pad_divide_by(mask, 16)
        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)
        (first_k, _, qf16, _, _) = self.encode_key(frame_idx)
        first_v = self.prop_net.encode_value(self.images[:, frame_idx], qf16, self.prob[1:, frame_idx])
        first_k = first_k.unsqueeze(2)
        self.do_pass(first_k, first_v, frame_idx, end_idx)