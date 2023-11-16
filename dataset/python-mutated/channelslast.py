from __future__ import annotations
from ..basics import *
from .fp16 import AMPMode, MixedPrecision
from torch.cuda.amp import GradScaler
__all__ = ['ChannelsLast']

class ChannelsLast(Callback):
    """Channels last training using PyTorch's Channels Last Memory Format (beta)"""
    order = -1

    def before_fit(self):
        if False:
            return 10
        self.learn.model.to(memory_format=torch.channels_last)

@patch
@delegates(GradScaler)
def to_channelslast(self: Learner, use_amp: bool=True, amp_mode: str | AMPMode=AMPMode.FP16, **kwargs):
    if False:
        while True:
            i = 10
    'Set `Learner` and inputs to `channels_last` format and float16 Mixed Precision by default'
    if use_amp and (not hasattr(self, 'mixed_precision')) and (not hasattr(self, 'channels_last')):
        return self.add_cbs([ChannelsLast(), MixedPrecision(amp_mode, **kwargs)])
    elif not hasattr(self, 'channels_last'):
        return self.add_cb(ChannelsLast())

@patch
def to_contiguous(self: Learner, to_fp32: bool=False):
    if False:
        i = 10
        return i + 15
    'Set `Learner` and inputs to `contiguous_format` (default format), optionally to single precision'
    self.model.to(memory_format=torch.contiguous_format)
    if to_fp32:
        return self.remove_cbs([ChannelsLast, MixedPrecision])
    else:
        return self.remove_cb(ChannelsLast)