import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
from torchmetrics.classification import MulticlassAccuracy

def finetune_pet_dataset(model_ft):
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
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 37)
    loss_ft = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    model = Trainer.compile(model_ft, loss_ft, optimizer_ft, metrics=[MulticlassAccuracy(num_classes=37)])
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return (model, train_dataset, val_dataset)
if __name__ == '__main__':
    model = resnet18(pretrained=True)
    (_, train_dataset, val_dataset) = finetune_pet_dataset(model)
    x = torch.stack([val_dataset[0][0], val_dataset[1][0]])
    model.eval()
    y_hat = model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)
    from bigdl.nano.pytorch import InferenceOptimizer
    q_model = InferenceOptimizer.quantize(model, calib_data=DataLoader(train_dataset, batch_size=32))
    with InferenceOptimizer.get_context(q_model):
        y_hat = q_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)
    InferenceOptimizer.save(q_model, './quantized_model')