from typing import Optional
from types import MethodType
from logging import warning
import torch
import pytorch_lightning as pl

def generate_channels_last_available(inputs):
    if False:
        return 10
    '\n    This function will generate a list of string to decide if the\n    elements of input can be converted to\n\n    channel_last: "channel_last"\n    channel_last_3d: "channel_last_3d"\n    no change: "original"\n    '
    if inputs is not None:
        if isinstance(inputs, torch.Tensor):
            inputs = tuple([inputs])
        channels_last_available = ['original'] * len(inputs)
        for (idx, input) in enumerate(inputs):
            try:
                input.to(memory_format=torch.channels_last)
                channels_last_available[idx] = 'channels_last'
            except Exception as _e:
                try:
                    input.to(memory_format=torch.channels_last_3d)
                    channels_last_available[idx] = 'channels_last_3d'
                except Exception as _e:
                    pass
    else:
        channels_last_available = []
    return channels_last_available

def apply_proper_channels_last(flag, input_item):
    if False:
        i = 10
        return i + 15
    '\n    This function will apply proper channes_last to\n    input item. flag has 3 possible values:\n\n    channel_last: "channel_last"\n    channel_last_3d: "channel_last_3d"\n    no change: "original"\n    '
    if flag == 'channels_last':
        return input_item.to(memory_format=torch.channels_last)
    if flag == 'channels_last_3d':
        return input_item.to(memory_format=torch.channels_last_3d)
    return input_item

def batch_call(func):
    if False:
        print('Hello World!')
    '\n    Decorator to extending hook of pl_module.\n\n    Extending behavior hook on_before_batch_transfer to convert data to channels_last\n    for each batch.\n    '

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if False:
            while True:
                i = 10

        def convert_channels_last(batch):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(batch, torch.Tensor) and batch.dim() == 4:
                batch = batch.to(memory_format=torch.channels_last)
            elif isinstance(batch, list) or isinstance(batch, tuple):
                batch = list(batch)
                for (index, t) in enumerate(batch):
                    batch[index] = convert_channels_last(t)
            return batch
        batch = func(batch, dataloader_idx)
        batch = convert_channels_last(batch)
        return batch
    return on_before_batch_transfer

class ChannelsLastCallback(pl.Callback):
    """Custom pl.Callback for converting model and data to channels_last."""

    def setup(self, trainer, pl_module, stage: Optional[str]=None) -> None:
        if False:
            return 10
        'Override hook setup to convert model to channels_last and wrap DataHook.'
        try:
            pl_module = pl_module.to(memory_format=torch.channels_last)
        except Exception as e:
            warning(f'Convert model to channels last failed,                     fall back to origin memory format. Exception msg: {e}')
            return super().setup(trainer, pl_module, stage)
        fn_old = getattr(pl_module, 'on_before_batch_transfer')
        fn = batch_call(fn_old)
        setattr(pl_module, 'on_before_batch_transfer_origin', fn_old)
        pl_module.on_before_batch_transfer = MethodType(fn, pl_module)
        return super().setup(trainer, pl_module, stage)

    def teardown(self, trainer, pl_module, stage: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Undo the changes to pl_module at end of fit, validate, tests, or predict.'
        if hasattr(pl_module, 'on_before_batch_transfer_origin'):
            setattr(pl_module, 'on_before_batch_transfer', pl_module.on_before_batch_transfer_origin)
            delattr(pl_module, 'on_before_batch_transfer_origin')
        return super().teardown(trainer, pl_module, stage)