"""
Auto-batch utils
"""
from copy import deepcopy
import numpy as np
import torch
from torch.cuda import amp
from utils.general import LOGGER, colorstr
from utils.torch_utils import profile

def check_train_batch_size(model, imgsz=640):
    if False:
        for i in range(10):
            print('nop')
    with amp.autocast():
        return autobatch(deepcopy(model).train(), imgsz)

def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    if False:
        while True:
            i = 10
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size
    gb = 1 << 30
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        y = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')
    y = [x[2] for x in y if x]
    batch_sizes = batch_sizes[:len(y)]
    p = np.polyfit(batch_sizes, y, deg=1)
    b = int((f * fraction - p[1]) / p[0])
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%)')
    return b