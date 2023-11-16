import os
import torch
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision.transforms import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from bigdl.nano.pytorch.vision.models import ImageClassifier

class Net(ImageClassifier):

    def __init__(self, backbone):
        if False:
            return 10
        super().__init__(backbone=backbone, num_classes=2)

    def configure_optimizers(self):
        if False:
            return 10
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002, amsgrad=True)
        return optimizer

def create_data_loader(root_dir, batch_size):
    if False:
        while True:
            i = 10
    dir_path = os.path.realpath(root_dir)
    data_transform = transforms.Compose([transforms.Resize(256), transforms.ColorJitter(), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.Resize(128), transforms.ToTensor()])
    catdogs = ImageFolder(dir_path, data_transform)
    val_num = len(catdogs) // 10
    train_num = len(catdogs) - val_num
    (train_set, val_set) = torch.utils.data.random_split(catdogs, [train_num, val_num])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return (train_loader, val_loader)

def train_torch_lightning(model, root_dir, batch_size):
    if False:
        for i in range(10):
            print('nop')
    (train_loader, val_loader) = create_data_loader(root_dir, batch_size)
    net = Net(model)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(net, train_loader)
    trainer.test(net, val_loader)
    print('pass')