import torch
import torchvision
from torch.distributed._tools import MemoryTracker

def run_one_model(net: torch.nn.Module, input: torch.Tensor):
    if False:
        return 10
    net.cuda()
    input = input.cuda()
    mem_tracker = MemoryTracker()
    mem_tracker.start_monitor(net)
    net.zero_grad(True)
    loss = net(input)
    if isinstance(loss, dict):
        loss = loss['out']
    loss.sum().backward()
    net.zero_grad(set_to_none=True)
    mem_tracker.stop()
    mem_tracker.summary()
    mem_tracker.show_traces()
run_one_model(torchvision.models.resnet34(), torch.rand(32, 3, 224, 224, device='cuda'))