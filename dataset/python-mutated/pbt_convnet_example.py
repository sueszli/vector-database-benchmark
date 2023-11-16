import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from ray.tune.examples.mnist_pytorch import train_func, test_func, ConvNet, get_data_loaders
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore

class PytorchTrainable(tune.Trainable):
    """Train a Pytorch ConvNet with Trainable and PopulationBasedTraining
       scheduler. The example reuse some of the functions in mnist_pytorch,
       and is a good demo for how to add the tuning function without
       changing the original training code.
    """

    def setup(self, config):
        if False:
            print('Hello World!')
        (self.train_loader, self.test_loader) = get_data_loaders()
        self.model = ConvNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.get('lr', 0.01), momentum=config.get('momentum', 0.9))

    def step(self):
        if False:
            return 10
        train_func(self.model, self.optimizer, self.train_loader)
        acc = test_func(self.model, self.test_loader)
        return {'mean_accuracy': acc}

    def save_checkpoint(self, checkpoint_dir):
        if False:
            while True:
                i = 10
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        if False:
            i = 10
            return i + 15
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        self.model.load_state_dict(torch.load(checkpoint_path))

    def reset_config(self, new_config):
        if False:
            while True:
                i = 10
        for param_group in self.optimizer.param_groups:
            if 'lr' in new_config:
                param_group['lr'] = new_config['lr']
            if 'momentum' in new_config:
                param_group['momentum'] = new_config['momentum']
        self.config = new_config
        return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    (args, _) = parser.parse_known_args()
    ray.init(num_cpus=2)
    datasets.MNIST('~/data', train=True, download=True)
    validate_save_restore(PytorchTrainable)
    scheduler = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=5, hyperparam_mutations={'lr': lambda : np.random.uniform(0.0001, 1), 'momentum': [0.8, 0.9, 0.99]})

    class CustomStopper(tune.Stopper):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.should_stop = False

        def __call__(self, trial_id, result):
            if False:
                i = 10
                return i + 15
            max_iter = 5 if args.smoke_test else 100
            if not self.should_stop and result['mean_accuracy'] > 0.96:
                self.should_stop = True
            return self.should_stop or result['training_iteration'] >= max_iter

        def stop_all(self):
            if False:
                return 10
            return self.should_stop
    stopper = CustomStopper()
    tuner = tune.Tuner(PytorchTrainable, run_config=train.RunConfig(name='pbt_test', stop=stopper, verbose=1, checkpoint_config=train.CheckpointConfig(checkpoint_score_attribute='mean_accuracy', checkpoint_frequency=5, num_to_keep=4)), tune_config=tune.TuneConfig(scheduler=scheduler, metric='mean_accuracy', mode='max', num_samples=4, reuse_actors=True), param_space={'lr': tune.uniform(0.001, 1), 'momentum': tune.uniform(0.001, 1)})
    results = tuner.fit()
    best_result = results.get_best_result()
    best_checkpoint = best_result.checkpoint