from tempfile import TemporaryDirectory
from pytest import mark
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from catalyst import dl, utils
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.data.sampler import BatchBalanceClassSampler
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import DATA_ROOT
if SETTINGS.ml_required:
    from sklearn.ensemble import RandomForestClassifier
TRAIN_EPOCH = 3
LR = 0.01
RANDOM_STATE = 42

def train_experiment(engine=None):
    if False:
        i = 10
        return i + 15
    with TemporaryDirectory() as logdir:
        utils.set_global_seed(RANDOM_STATE)
        train_data = MNIST(DATA_ROOT, train=True)
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = BatchBalanceClassSampler(train_labels, num_classes=10, num_samples=4)
        train_loader = DataLoader(train_data, batch_sampler=train_sampler)
        valid_dataset = MNIST(root=DATA_ROOT, train=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=32)
        test_dataset = MNIST(root=DATA_ROOT, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32)
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 16), nn.LeakyReLU(inplace=True))
        optimizer = Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
        sampler_inbatch = HardTripletsSampler(norm_required=False)
        criterion = TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)

        class CustomRunner(dl.SupervisedRunner):

            def handle_batch(self, batch) -> None:
                if False:
                    i = 10
                    return i + 15
                (images, targets) = (batch['features'].float(), batch['targets'].long())
                features = self.model(images)
                self.batch = {'embeddings': features, 'targets': targets}
        callbacks = [dl.ControlFlowCallbackWrapper(dl.CriterionCallback(input_key='embeddings', target_key='targets', metric_key='loss'), loaders='train'), dl.SklearnModelCallback(feature_key='embeddings', target_key='targets', train_loader='train', valid_loaders=['valid', 'infer'], model_fn=RandomForestClassifier, predict_method='predict_proba', predict_key='sklearn_predict', random_state=RANDOM_STATE, n_estimators=50), dl.ControlFlowCallbackWrapper(dl.AccuracyCallback(target_key='targets', input_key='sklearn_predict', topk=(1, 3)), loaders=['valid', 'infer'])]
        runner = CustomRunner(input_key='features', output_key='embeddings')
        runner.train(engine=engine, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, callbacks=callbacks, loaders={'train': train_loader, 'valid': valid_loader, 'infer': test_loader}, verbose=False, valid_loader='valid', valid_metric='accuracy01', minimize_valid_metric=False, num_epochs=TRAIN_EPOCH, logdir=logdir)
        best_accuracy = max((epoch_metrics['infer']['accuracy01'] for epoch_metrics in runner.experiment_metrics.values()))
        assert best_accuracy > 0.8
requirements_satisfied = SETTINGS.ml_required

@mark.skipif(not requirements_satisfied, reason='catalyst[ml] and catalyst[cv] required')
def test_run_on_cpu():
    if False:
        print('Hello World!')
    train_experiment(dl.CPUEngine())

@mark.skipif(not all([requirements_satisfied, IS_CUDA_AVAILABLE]), reason='CUDA device is not available')
def test_run_on_torch_cuda0():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.GPUEngine())

@mark.skipif(not all([requirements_satisfied, IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2]), reason='No CUDA>=2 found')
def test_run_on_torch_dp():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.DataParallelEngine())

@mark.skipif(not all([requirements_satisfied, IS_CUDA_AVAILABLE and SETTINGS.amp_required]), reason='No CUDA or AMP found')
def test_run_on_amp():
    if False:
        i = 10
        return i + 15
    train_experiment(dl.GPUEngine(fp16=True))

@mark.skipif(not all([requirements_satisfied, IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2 and SETTINGS.amp_required]), reason='No CUDA>=2 or AMP found')
def test_run_on_amp_dp():
    if False:
        return 10
    train_experiment(dl.DataParallelEngine(fp16=True))