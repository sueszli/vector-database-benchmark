import os
import json
import torch
import time
from bigdl.nano.pytorch.vision.datasets import ImageFolder
from bigdl.nano.pytorch.vision.transforms import transforms
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from bigdl.nano.pytorch.vision.models import resnet50
from bigdl.nano.pytorch.vision.models import ImageClassifier
from bigdl.nano.pytorch import Trainer
DATA_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
parser = argparse.ArgumentParser(description='PyTorch Cat Dog')
parser.add_argument('--name', default='PyTorch Cat Dog', type=str)
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=5, type=int, help='epoch number')
parser.add_argument('--nano_data', default=False, type=bool)
parser.add_argument('--use_ipex', default=False, type=bool)
parser.add_argument('--nproc', default=1, type=int)

class Classifier(ImageClassifier):

    def __init__(self):
        if False:
            return 10
        backbone = resnet50(pretrained=False, include_top=False, freeze=False)
        super().__init__(backbone=backbone, num_classes=2)

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        return torch.optim.Adam(self.parameters(), lr=0.002)

def create_data_loader(root_dir, batch_size, nproc):
    if False:
        return 10
    dir_path = os.path.realpath(root_dir)
    data_transform = transforms.Compose([transforms.Resize(256), transforms.ColorJitter(), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.Resize(128), transforms.ToTensor()])
    catdogs = ImageFolder(dir_path, data_transform)
    dataset_size = len(catdogs)
    if 'SANITY_CHECK' in os.environ and os.environ['SANITY_CHECK'] == '1':
        train_size = 2 * nproc * batch_size
        val_size = 2 * nproc * batch_size
        test_size = dataset_size - train_size - val_size
        (train_set, val_set, _) = torch.utils.data.random_split(catdogs, [train_size, val_size, test_size])
    else:
        train_split = 0.8
        train_size = int(dataset_size * train_split)
        val_size = dataset_size - train_size
        (train_set, val_set) = torch.utils.data.random_split(catdogs, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    size = (train_size, val_size)
    return (train_loader, val_loader, size)

def _main_process(args, train_loader, val_loader, train_size):
    if False:
        while True:
            i = 10
    classifier = Classifier()
    trainer = Trainer(max_epochs=args.epochs, use_ipex=args.use_ipex, num_processes=args.nproc)
    train_start = time.time()
    trainer.fit(classifier, train_loader)
    train_end = time.time()
    trainer.test(classifier, val_loader)
    output = json.dumps({'config': args.name, 'train_time': train_end - train_start, 'train_throughput': train_size * args.epochs / (train_end - train_start)})
    print(f'>>>{output}<<<')
    return

def main():
    if False:
        return 10
    print('Nano Pytorch cat-vs-dog example')
    args = parser.parse_args()
    if args.nano_data:
        from bigdl.nano.pytorch.vision.datasets import ImageFolder
        from bigdl.nano.pytorch.vision import transforms
    else:
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
    download_and_extract_archive(url=DATA_URL, download_root='data')
    (train_loader, val_loader, (train_size, _)) = create_data_loader('data', args.batch_size, args.nproc)
    _main_process(args, train_loader, val_loader, train_size)
if __name__ == '__main__':
    main()