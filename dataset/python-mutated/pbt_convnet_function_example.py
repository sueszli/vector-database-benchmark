import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import test_func, ConvNet, get_data_loaders
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining

def train_convnet(config):
    if False:
        i = 10
        return i + 15
    step = 0
    (train_loader, test_loader) = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(model.parameters(), lr=config.get('lr', 0.01), momentum=config.get('momentum', 0.9))
    if train.get_checkpoint():
        print('Loading from checkpoint.')
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            path = os.path.join(loaded_checkpoint_dir, 'checkpoint.pt')
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
    while True:
        ray.tune.examples.mnist_pytorch.train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)
        checkpoint = None
        if step % 5 == 0:
            os.makedirs('my_model', exist_ok=True)
            torch.save({'step': step, 'model': model.state_dict()}, 'my_model/checkpoint.pt')
            checkpoint = Checkpoint.from_directory('my_model')
        step += 1
        train.report({'mean_accuracy': acc}, checkpoint=checkpoint)

def eval_best_model(results: tune.ResultGrid):
    if False:
        return 10
    'Test the best model given output of tuner.fit().'
    with results.get_best_result().checkpoint.as_directory() as best_checkpoint_path:
        best_model = ConvNet()
        best_checkpoint = torch.load(os.path.join(best_checkpoint_path, 'checkpoint.pt'))
        best_model.load_state_dict(best_checkpoint['model'])
        test_acc = test_func(best_model, get_data_loaders()[1])
        print('best model accuracy: ', test_acc)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    (args, _) = parser.parse_known_args()
    scheduler = PopulationBasedTraining(time_attr='training_iteration', perturbation_interval=5, hyperparam_mutations={'lr': lambda : np.random.uniform(0.0001, 1), 'momentum': [0.8, 0.9, 0.99]})

    class CustomStopper(tune.Stopper):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.should_stop = False

        def __call__(self, trial_id, result):
            if False:
                while True:
                    i = 10
            max_iter = 5 if args.smoke_test else 100
            if not self.should_stop and result['mean_accuracy'] > 0.96:
                self.should_stop = True
            return self.should_stop or result['training_iteration'] >= max_iter

        def stop_all(self):
            if False:
                while True:
                    i = 10
            return self.should_stop
    stopper = CustomStopper()
    tuner = tune.Tuner(train_convnet, run_config=train.RunConfig(name='pbt_test', stop=stopper, verbose=1, checkpoint_config=train.CheckpointConfig(checkpoint_score_attribute='mean_accuracy', num_to_keep=4)), tune_config=tune.TuneConfig(scheduler=scheduler, metric='mean_accuracy', mode='max', num_samples=4), param_space={'lr': tune.uniform(0.001, 1), 'momentum': tune.uniform(0.001, 1)})
    results = tuner.fit()
    eval_best_model(results)