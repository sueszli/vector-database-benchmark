import argparse
import torch
import torch.multiprocessing as mp
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule
from pyro.optim import Adam, HorovodOptimizer

class Model(PyroModule):

    def __init__(self, size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.size = size

    def forward(self, covariates, data=None):
        if False:
            print('Hello World!')
        coeff = pyro.sample('coeff', dist.Normal(0, 1))
        bias = pyro.sample('bias', dist.Normal(0, 1))
        scale = pyro.sample('scale', dist.LogNormal(0, 1))
        with pyro.plate('data', self.size, len(covariates)):
            loc = bias + coeff * covariates
            return pyro.sample('obs', dist.Normal(loc, scale), obs=data)

def main(args):
    if False:
        for i in range(10):
            print('nop')
    pyro.set_rng_seed(args.seed)
    model = Model(args.size)
    covariates = torch.randn(args.size)
    data = model(covariates)
    guide = AutoNormal(model)
    if args.horovod:
        import horovod.torch as hvd
        hvd.init()
        torch.set_num_threads(1)
        if args.cuda:
            torch.cuda.set_device(hvd.local_rank())
    if args.cuda:
        torch.set_default_device('cuda')
    device = torch.tensor(0).device
    if args.horovod:
        guide(covariates[:1], data[:1])
        hvd.broadcast_parameters(guide.state_dict(), root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    elbo = Trace_ELBO()
    optim = Adam({'lr': args.learning_rate})
    if args.horovod:
        optim = HorovodOptimizer(optim)
    dataset = torch.utils.data.TensorDataset(covariates, data)
    if args.horovod:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, hvd.size(), hvd.rank())
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    config = {'batch_size': args.batch_size, 'sampler': sampler}
    if args.cuda:
        config['num_workers'] = 1
        config['pin_memory'] = True
        if hasattr(mp, '_supports_context') and mp._supports_context and ('forkserver' in mp.get_all_start_methods()):
            config['multiprocessing_context'] = 'forkserver'
    dataloader = torch.utils.data.DataLoader(dataset, **config)
    svi = SVI(model, guide, optim, elbo)
    for epoch in range(args.num_epochs):
        if args.horovod:
            sampler.set_epoch(epoch)
        for (step, (covariates_batch, data_batch)) in enumerate(dataloader):
            loss = svi.step(covariates_batch.to(device), data_batch.to(device))
            if args.horovod:
                loss = torch.tensor(loss)
                loss = hvd.allreduce(loss, 'loss')
                loss = loss.item()
                if step % 100 == 0 and hvd.rank() == 0:
                    print('epoch {} step {} loss = {:0.4g}'.format(epoch, step, loss))
            elif step % 100 == 0:
                print('epoch {} step {} loss = {:0.4g}'.format(epoch, step, loss))
    if args.horovod:
        hvd.shutdown()
        if hvd.rank() != 0:
            return
    if args.outfile:
        print('saving to {}'.format(args.outfile))
        torch.save({'model': model, 'guide': guide}, args.outfile)
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='Distributed training via Horovod')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-s', '--size', default=1000000, type=int)
    parser.add_argument('-b', '--batch-size', default=100, type=int)
    parser.add_argument('-n', '--num-epochs', default=10, type=int)
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--horovod', action='store_true', default=True)
    parser.add_argument('--no-horovod', action='store_false', dest='horovod')
    parser.add_argument('--seed', default=20200723, type=int)
    args = parser.parse_args()
    main(args)