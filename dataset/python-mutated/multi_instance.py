from typing import Union, List
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.utils.common import invalidInputError, invalidOperationError
from bigdl.nano.pytorch.model import AcceleratedLightningModule

class _MultiInstanceModel(torch.nn.Module):

    def __init__(self, model, ps, send_queue, recv_queue, next_idx):
        if False:
            print('Hello World!')
        super().__init__()
        self.model = model
        self.ps = ps
        self.p_num = len(ps) if ps is not None else 0
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.next_idx = next_idx

    def forward(self, input_data: Union[DataLoader, List]) -> List:
        if False:
            return 10
        if isinstance(input_data, (DataLoader, list)):
            length = len(input_data)
        else:
            invalidInputError(False, 'The input should be a DataLoader or a list of input batchs')
        if self.ps is None:
            from bigdl.nano.pytorch import InferenceOptimizer
            with InferenceOptimizer.get_context(self.model):
                outputs = [self.model(inputs) for inputs in input_data]
            return outputs
        else:
            if isinstance(input_data, DataLoader) and (not isinstance(input_data.dataset, TensorDataset)):
                for idx in range(self.p_num):
                    self.send_queue.put(input_data)
            else:
                for idx_and_batch in enumerate(input_data):
                    self.send_queue.put(idx_and_batch)
            outputs = [None] * length
            for _i in range(length):
                (idx, output) = self.recv_queue.get()
                invalidOperationError(not isinstance(output, Exception), f'forward error: {output}\n')
                outputs[idx] = output
            with self.next_idx.get_lock():
                self.next_idx.value = 0
        return outputs

def _multi_instance_helper(model, recv_queue, send_queue, next_idx):
    if False:
        print('Hello World!')
    from bigdl.nano.pytorch import InferenceOptimizer
    with InferenceOptimizer.get_context(model):
        while True:
            try:
                args = recv_queue.get()
                if isinstance(args, DataLoader):
                    dataset = args.dataset
                    batch_size = args.batch_size
                    length = len(args)
                    idx = get_next_idx(next_idx)
                    while idx < length:
                        batch = [dataset[i] for i in range(idx * batch_size, (idx + 1) * batch_size)]
                        inputs = args.collate_fn(batch)
                        new_idx = get_next_idx(next_idx)
                        inference(idx, inputs, model, send_queue)
                        idx = new_idx
                else:
                    (idx, inputs) = args
                    inference(idx, inputs, model, send_queue)
            except Exception as e:
                send_queue.put((idx, e))

def inference(idx, inputs, model, send_queue):
    if False:
        while True:
            i = 10
    if isinstance(inputs, tuple):
        output = model(*inputs)
    else:
        output = model(inputs)
    send_queue.put((idx, output))

def get_next_idx(next_idx):
    if False:
        print('Hello World!')
    with next_idx.get_lock():
        idx = next_idx.value
        next_idx.value += 1
    return idx