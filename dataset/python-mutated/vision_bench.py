from typing import Callable
import torch
from deepchecks.core.errors import DeepchecksBaseError
from deepchecks.vision import SingleDatasetCheck, TrainTestCheck
from deepchecks.vision.datasets.classification import mnist_torch as mnist
from deepchecks.vision.datasets.detection import coco_torch as coco
from deepchecks.vision.vision_data import VisionData
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_static_predictions(train: VisionData, test: VisionData, model):
    if False:
        while True:
            i = 10
    static_preds = []
    for vision_data in [train, test]:
        if vision_data is not None:
            static_pred = {}
            for (i, batch) in enumerate(vision_data):
                predictions = vision_data.infer_on_batch(batch, model, device)
                indexes = list(vision_data.data_loader.batch_sampler)[i]
                static_pred.update(dict(zip(indexes, predictions)))
        else:
            static_pred = None
        static_preds.append(static_pred)
    (train_preds, tests_preds) = static_preds
    return (train_preds, tests_preds)

def run_check_fn(check_class) -> Callable:
    if False:
        print('Hello World!')

    def run(self, cache, dataset_name):
        if False:
            while True:
                i = 10
        (train_ds, test_ds, train_pred, test_pred) = cache[dataset_name]
        check = check_class()
        try:
            if isinstance(check, SingleDatasetCheck):
                check.run(train_ds, train_predictions=train_pred, device=device)
            elif isinstance(check, TrainTestCheck):
                check.run(train_ds, test_ds, train_predictions=train_pred, test_predictions=test_pred, device=device)
        except DeepchecksBaseError:
            pass
    return run

def setup_mnist():
    if False:
        for i in range(10):
            print('nop')
    mnist_model = mnist.load_model()
    train_ds = mnist.load_dataset(train=True, object_type='VisionData')
    test_ds = mnist.load_dataset(train=False, object_type='VisionData')
    (train_preds, tests_preds) = create_static_predictions(train_ds, test_ds, mnist_model)
    return (train_ds, test_ds, train_preds, tests_preds)

def setup_coco():
    if False:
        while True:
            i = 10
    coco_model = coco.load_model()
    train_ds = coco.load_dataset(train=True, object_type='VisionData')
    test_ds = coco.load_dataset(train=False, object_type='VisionData')
    (train_preds, tests_preds) = create_static_predictions(train_ds, test_ds, coco_model)
    return (train_ds, test_ds, train_preds, tests_preds)

class BenchmarkVision:
    timeout = 120
    params = ['mnist', 'coco']
    param_names = ['dataset_name']

    def setup_cache(self):
        if False:
            return 10
        cache = {}
        cache['mnist'] = setup_mnist()
        cache['coco'] = setup_coco()
        return cache