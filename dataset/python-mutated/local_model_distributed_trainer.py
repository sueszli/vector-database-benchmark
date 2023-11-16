import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from bigdl.chronos.model.tcmf.local_model import TemporalConvNet
import ray
import horovod.torch as hvd

def get_tcmf_data_loader(config):
    if False:
        print('Hello World!')
    from bigdl.chronos.model.tcmf.data_loader import TCMFDataLoader
    tcmf_data_loader = TCMFDataLoader(Ymat=ray.get(config['Ymat_id']), vbsize=config['vbsize'], hbsize=config['hbsize'], end_index=config['end_index'], val_len=config['val_len'], covariates=ray.get(config['covariates_id']), Ycov=ray.get(config['Ycov_id']))
    return tcmf_data_loader

class TcmfTrainDatasetDist(torch.utils.data.IterableDataset):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super(TcmfTrainDatasetDist).__init__()
        self.tcmf_data_loader = get_tcmf_data_loader(config)
        self.last_epoch = 0

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        while self.tcmf_data_loader.epoch == self.last_epoch:
            yield self.get_next_batch()
        self.last_epoch += 1

    def get_next_batch(self):
        if False:
            print('Hello World!')
        (inp, out, _, _) = self.tcmf_data_loader.next_batch()
        if dist.is_initialized():
            num_workers = dist.get_world_size()
            per_worker = inp.shape[0] // num_workers
            inp_parts = torch.split(inp, per_worker)
            out_parts = torch.split(out, per_worker)
            worker_id = dist.get_rank()
            inp = inp_parts[worker_id]
            out = out_parts[worker_id]
        return (inp, out)

class TcmfTrainDatasetHorovod(torch.utils.data.IterableDataset):

    def __init__(self, config):
        if False:
            return 10
        super(TcmfTrainDatasetHorovod).__init__()
        self.tcmf_data_loader = get_tcmf_data_loader(config)
        self.last_epoch = 0

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        while self.tcmf_data_loader.epoch == self.last_epoch:
            yield self.get_next_batch()
        self.last_epoch += 1

    def get_next_batch(self):
        if False:
            while True:
                i = 10
        (inp, out, _, _) = self.tcmf_data_loader.next_batch()
        try:
            num_workers = hvd.size()
            per_worker = inp.shape[0] // num_workers
            inp_parts = torch.split(inp, per_worker)
            out_parts = torch.split(out, per_worker)
            worker_id = hvd.rank()
            inp = inp_parts[worker_id]
            out = out_parts[worker_id]
        except:
            pass
        return (inp, out)

class TcmfValDataset(torch.utils.data.IterableDataset):

    def __init__(self, config):
        if False:
            print('Hello World!')
        super(TcmfValDataset).__init__()
        self.tcmf_data_loader = get_tcmf_data_loader(config)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        (inp, out, _, _) = self.tcmf_data_loader.supply_test()
        yield (inp, out)

def data_creator(config):
    if False:
        return 10
    train_loader = DataLoader(TcmfTrainDatasetDist(config), batch_size=None)
    val_loader = DataLoader(TcmfValDataset(config), batch_size=None)
    return (train_loader, val_loader)

def train_data_creator(config, batch_size):
    if False:
        while True:
            i = 10
    return DataLoader(TcmfTrainDatasetHorovod(config), batch_size=None)

def val_data_creator(config, batch_size):
    if False:
        i = 10
        return i + 15
    return DataLoader(TcmfValDataset(config), batch_size=None)

def tcmf_loss(out, target):
    if False:
        i = 10
        return i + 15
    criterion = nn.L1Loss()
    return criterion(out, target) / torch.abs(target.data).mean()

def loss_creator(config):
    if False:
        i = 10
        return i + 15
    return tcmf_loss

def optimizer_creator(model, config):
    if False:
        for i in range(10):
            print('nop')
    'Returns optimizer.'
    return optim.Adam(model.parameters(), lr=config['lr'])

def model_creator(config):
    if False:
        print('Hello World!')
    return TemporalConvNet(num_inputs=config['num_inputs'], num_channels=config['num_channels'], kernel_size=config['kernel_size'], dropout=config['dropout'], init=True)

def train_yseq_hvd(workers_per_node, epochs, **config):
    if False:
        for i in range(10):
            print('nop')
    from bigdl.orca.learn.pytorch import Estimator
    estimator = Estimator.from_torch(model=model_creator, optimizer=optimizer_creator, loss=loss_creator, workers_per_node=workers_per_node, config=config, backend='horovod')
    stats = estimator.fit(train_data_creator, epochs=epochs)
    for s in stats:
        for (k, v) in s.items():
            print(f'{k}: {v}')
    val_stats = estimator.evaluate(val_data_creator)
    val_loss = val_stats['val_loss']
    yseq = estimator.get_model()
    estimator.shutdown()
    return (yseq, val_loss)