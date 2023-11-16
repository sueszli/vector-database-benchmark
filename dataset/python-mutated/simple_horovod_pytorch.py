from __future__ import print_function
import argparse
import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.horovod import HorovodRayRunner

def run_horovod():
    if False:
        print('Hello World!')
    import urllib
    try:

        class AppURLopener(urllib.FancyURLopener):
            version = 'Mozilla/5.0'
        urllib._urlopener = AppURLopener()
    except AttributeError:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 43
    log_interval = 10
    fp16_allreduce = False
    use_adasum = False
    hvd.init()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    kwargs = {}
    train_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs)

    class Net(nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)
    model = Net()
    lr_scaler = hvd.size() if not use_adasum else 1
    optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler, momentum=momentum)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression, op=hvd.Adasum if use_adasum else hvd.Average)

    def train(epoch):
        if False:
            print('Hello World!')
        model.train()
        train_sampler.set_epoch(epoch)
        for (batch_idx, (data, target)) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_sampler), 100.0 * batch_idx / len(train_loader), loss.item()))

    def metric_average(val, name):
        if False:
            i = 10
            return i + 15
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        if False:
            print('Hello World!')
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        for (data, target) in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, 100.0 * test_accuracy))
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default='local', help='The mode for the Spark cluster.')
parser.add_argument('--slave_num', type=int, default=2, help='The number of slave nodes to be used in the cluster.You can change it depending on your own cluster setting.')
parser.add_argument('--cores', type=int, default=8, help='The number of cpu cores you want to use on each node. You can change it depending on your own cluster setting.')
parser.add_argument('--memory', type=str, default='10g', help="The size of slave(executor)'s memory you want to use.You can change it depending on your own cluster setting.")
if __name__ == '__main__':
    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == 'local' else args.slave_num
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores, num_nodes=num_nodes, memory=args.memory)
    runner = HorovodRayRunner(OrcaRayContext.get())
    runner.run(func=run_horovod)
    stop_orca_context()