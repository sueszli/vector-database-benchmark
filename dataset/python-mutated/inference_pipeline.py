import torch
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from _finetune import MilestonesFinetuning, TransferLearningModel, CatDogImageDataModule
from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch import InferenceOptimizer
if __name__ == '__main__':
    milestones: tuple = (1, 2)
    trainer = Trainer(max_epochs=2, callbacks=[MilestonesFinetuning(milestones)])
    model = TransferLearningModel(milestones=milestones)
    datamodule = CatDogImageDataModule()
    trainer.fit(model, datamodule)

    def accuracy(pred, target):
        if False:
            print('Hello World!')
        pred = torch.sigmoid(pred)
        target = target.view((-1, 1)).type_as(pred).int()
        return multiclass_accuracy(pred, target, num_classes=2)
    optimizer = InferenceOptimizer()
    optimizer.optimize(model=model, training_data=datamodule.train_dataloader(batch_size=1), validation_data=datamodule.val_dataloader(limit_num_samples=160), metric=accuracy, direction='max', thread_num=1, latency_sample_num=10)
    (acc_model, option) = optimizer.get_best_model(accuracy_criterion=0.05)
    print('When accuracy drop less than 5%, the model with minimal latency is: ', option)
    with InferenceOptimizer.get_context(acc_model):
        x_input = next(iter(datamodule.train_dataloader(batch_size=1)))[0]
        output = acc_model(x_input)