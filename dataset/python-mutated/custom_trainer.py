import torch
from ray import train
from ray.train.trainer import BaseTrainer

class MyPytorchTrainer(BaseTrainer):

    def setup(self):
        if False:
            return 10
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def training_loop(self):
        if False:
            while True:
                i = 10
        dataset = self.datasets['train']
        loss_fn = torch.nn.MSELoss()
        for epoch_idx in range(10):
            loss = 0
            num_batches = 0
            for batch in dataset.iter_torch_batches(dtypes=torch.float):
                (X, y) = (torch.unsqueeze(batch['x'], 1), batch['y'])
                pred = self.model(X)
                batch_loss = loss_fn(pred, y)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                loss += batch_loss.item()
                num_batches += 1
            loss /= num_batches
            train.report({'loss': loss, 'epoch': epoch_idx})
import ray
train_dataset = ray.data.from_items([{'x': i, 'y': i} for i in range(3)])
my_trainer = MyPytorchTrainer(datasets={'train': train_dataset})
result = my_trainer.fit()