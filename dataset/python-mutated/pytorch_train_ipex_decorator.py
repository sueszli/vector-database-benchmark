import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm
from bigdl.nano.pytorch import nano

class MyPytorchModule(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 37)

    def forward(self, x):
        if False:
            return 10
        return self.model(x)

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

@nano(num_processes=2, distributed_backend='subprocess')
def trainingggg_loop(model, optimizer, train_loader, val_loader, num_epochs, loss_func):
    if False:
        i = 10
        return i + 15
    for epoch in range(num_epochs):
        model.train()
        (train_loss, num) = (0, 0)
        with tqdm(train_loader, unit='batch') as tepoch:
            for (data, target) in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                loss_value = loss.sum()
                train_loss += loss_value
                num += 1
                tepoch.set_postfix(loss=loss_value)
            print(f'Train Epoch: {epoch}, avg_loss: {train_loss / num}')
if __name__ == '__main__':
    model = MyPytorchModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    loss_func = torch.nn.CrossEntropyLoss()
    (train_loader, val_loader) = create_dataloaders()
    trainingggg_loop(model, optimizer, train_loader, val_loader, num_epochs=5, loss_func=loss_func)