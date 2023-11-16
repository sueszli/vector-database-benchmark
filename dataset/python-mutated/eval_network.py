import torch
import torch.nn as nn
from modelscope.models.cv.video_object_segmentation.modules import KeyEncoder, KeyProjection, MemCrompress, ValueEncoder
from modelscope.models.cv.video_object_segmentation.network import Decoder

class RDE_VOS(nn.Module):

    def __init__(self, repeat=0):
        if False:
            return 10
        super().__init__()
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder()
        self.key_proj = KeyProjection(1024, keydim=64)
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder = Decoder()
        self.mem_compress = MemCrompress(repeat=repeat)

    def encode_value(self, frame, kf16, masks):
        if False:
            return 10
        (k, _, h, w) = masks.shape
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        if k != 1:
            others = torch.cat([torch.sum(masks[[j for j in range(k) if i != j]], dim=0, keepdim=True) for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)
        f16 = self.value_encoder(frame, kf16.repeat(k, 1, 1, 1), masks, others)
        return f16.unsqueeze(2)

    def encode_key(self, frame):
        if False:
            i = 10
            return i + 15
        (f16, f8, f4) = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)
        return (k16, f16_thin, f16, f8, f4)

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16):
        if False:
            for i in range(10):
                print('nop')
        k = mem_bank.num_objects
        readout_mem = mem_bank.match_memory(qk16)
        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)
        return torch.sigmoid(self.decoder(qv16, qf8, qf4))