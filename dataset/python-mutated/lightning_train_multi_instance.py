import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader
import torch
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
import pytorch_lightning as pl

class MyLightningModule(pl.LightningModule):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 37)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if False:
            return 10
        (x, y) = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        if False:
            print('Hello World!')
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

def create_dataloaders():
    if False:
        return 10
    train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.5, hue=0.3), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = OxfordIIITPet(root='/tmp/data', transform=train_transform, download=True)
    val_dataset = OxfordIIITPet(root='/tmp/data', transform=val_transform)
    indices = torch.randperm(len(train_dataset))
    val_size = len(train_dataset) // 4
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    return (train_dataloader, val_dataloader)
if __name__ == '__main__':
    model = MyLightningModule()
    (train_loader, val_loader) = create_dataloaders()
    trainer = Trainer(max_epochs=5, num_processes=2)
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.validate(model, dataloaders=val_loader)