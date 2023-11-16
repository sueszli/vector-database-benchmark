import torch
from copy import deepcopy
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from bigdl.nano.pytorch.vision.models import ImageClassifier
num_classes = 10
data_transform = transforms.Compose([transforms.Resize(256), transforms.ColorJitter(), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.Resize(128), transforms.ToTensor()])
test_data_transform = transforms.Compose([transforms.Resize(256), transforms.ColorJitter(), transforms.Resize(128), transforms.ToTensor()])

class Net(ImageClassifier):

    def __init__(self, backbone):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(backbone=backbone, num_classes=num_classes)

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)
        return optimizer

def create_data_loader(dir, batch_size, num_workers, transform, subset=50, shuffle=True, sampler=False):
    if False:
        return 10
    train_set = CIFAR10(root=dir, train=True, download=True, transform=transform)
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)
    if sampler:
        sampler_set = SequentialSampler(train_subset)
        data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler_set)
    else:
        data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def create_test_data_loader(dir, batch_size, num_workers, transform, subset=50):
    if False:
        print('Hello World!')
    '\n    This function is to create a fixed dataset without any randomness\n    '
    train_set = CIFAR10(root=dir, train=False, download=True, transform=transform)
    mask = list(range(0, len(train_set), subset))
    train_subset = torch.utils.data.Subset(train_set, mask)
    data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader

def train_with_linear_top_layer(model_without_top, batch_size, num_workers, data_dir, use_ipex=False):
    if False:
        while True:
            i = 10
    model = Net(model_without_top)
    train_torch_lightning(model, batch_size, num_workers, data_dir, use_ipex=use_ipex)

def train_torch_lightning(model, batch_size, num_workers, data_dir, use_ipex=False):
    if False:
        print('Hello World!')
    orig_parameters = deepcopy(model.state_dict())
    orig_parameters_list = deepcopy(list(model.named_parameters()))
    train_loader = create_data_loader(data_dir, batch_size, num_workers, data_transform)
    from bigdl.nano.pytorch import Trainer
    trainer = Trainer(max_epochs=1, use_ipex=use_ipex)
    trainer.fit(model, train_loader)
    trained_parameters = model.state_dict()
    for i in range(len(orig_parameters_list)):
        (name, para) = orig_parameters_list[i]
        para1 = orig_parameters[name]
        para2 = trained_parameters[name]
        if name == 'model.1.bias' or name == 'model.1.weight' or name == 'new_classifier.1.bias' or (name == 'new_classifier.1.weight'):
            if torch.all(torch.eq(para1, para2)):
                raise Exception('Parameter ' + name + ' remains the same after training.')
        elif not torch.all(torch.eq(para1, para2)):
            raise Exception(name + ' freeze failed.')
    print('pass')