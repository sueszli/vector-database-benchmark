"""Module representing the MNIST dataset in tensorflow."""
try:
    from tensorflow import keras
except ImportError as error:
    raise ImportError('tensorflow is not installed. Please install tensorflow>=2.0.0 in order to use the selected dataset.') from error
import logging
import pathlib
import pickle
import typing as t
import numpy as np
from deepchecks.vision.utils.test_utils import hash_image
from deepchecks.vision.vision_data import VisionData
__all__ = ['load_dataset', 'load_model']
MNIST_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'mnist'
MODEL_PATH = MNIST_DIR / 'mnist_tf_model.ckpt'
MODEL_SAVED_PATH = MNIST_DIR / 'mnist_tf_model.ckpt.index'
LOGGER = logging.getLogger(__name__)

def load_dataset(train: bool=True, with_predictions: bool=True, batch_size: t.Optional[int]=None, shuffle: bool=False, n_samples: int=None, object_type='VisionData') -> VisionData:
    if False:
        for i in range(10):
            print('nop')
    "Return MNIST VisionData, containing prediction produced by a simple fully connected model.\n\n    Model and data are taken from https://www.tensorflow.org/tutorials/quickstart/beginner.\n\n    Parameters\n    ----------\n    train : bool, default : True\n        Train or Test dataset\n    with_predictions : bool, default : True\n        Whether the returned VisonData should contain predictions\n    batch_size: int, optional\n        how many samples per batch to load\n    shuffle : bool , default : False\n        To reshuffled data at every epoch or not.\n    n_samples : int, optional\n        Number of samples to load. Return the first n_samples if shuffle is False otherwise selects n_samples at random.\n        If None, returns all samples.\n    object_type : str, default : 'VisionData'\n        Kept for compatibility with torch datasets. Not used.\n    Returns\n    -------\n    :obj:`deepchecks.vision.VisionData`\n    "
    if object_type != 'VisionData':
        raise ValueError('only VisionData is supported for MNIST dataset')
    batch_size = batch_size or (64 if train else 1000)
    if with_predictions:
        model = load_model()
    else:
        model = None
    return VisionData(batch_loader=mnist_generator(shuffle, batch_size, train, n_samples, model), task_type='classification', dataset_name=f"mnist {('train' if train else 'test')}", reshuffle_data=False)

def mnist_generator(shuffle: bool=False, batch_size: int=64, train: bool=True, n_samples: int=None, model=None) -> t.Generator:
    if False:
        return 10
    'Generate an MNIST dataset.\n\n    Parameters\n    ----------\n    batch_size: int, optional\n        how many samples per batch to load\n    train : bool, default : True\n        Train or Test dataset\n    n_samples : int, optional\n        Number of samples to load.\n    shuffle : bool , default : False\n        whether to shuffle the data or not.\n    model : MockModel, optional\n        Model to use for predictions\n\n    Returns\n    -------\n    :obj:`t.Generator`\n    '
    (images, labels) = load_mnist_data(train, n_samples=n_samples, shuffle=shuffle)
    for i in range(0, len(images), batch_size):
        return_dict = {'images': images[i:i + batch_size], 'labels': labels[i:i + batch_size]}
        if model is not None:
            return_dict.update({'predictions': model(return_dict['images'])})
        return_dict['images'] = return_dict['images'] * 255.0
        yield return_dict

def load_mnist_data(train: bool=True, n_samples: int=None, shuffle: bool=False) -> t.Tuple[np.array, np.array]:
    if False:
        for i in range(10):
            print('nop')
    'Load MNIST dataset.\n\n    Parameters\n    ----------\n    train : bool, default : True\n        Train or Test dataset\n    n_samples : int, optional\n        Number of samples to load.\n    shuffle : bool , default : False\n        whether to shuffle the data or not.\n    Returns\n    -------\n    Tuple[np.ndarray, np.ndarray]\n    '
    if train:
        ((images, labels), _) = keras.datasets.mnist.load_data()
    else:
        (_, (images, labels)) = keras.datasets.mnist.load_data()
    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]
        del indices
    if n_samples is not None:
        images = images[:n_samples]
        labels = labels[:n_samples]
    images = images / 255.0
    images = np.expand_dims(images, axis=-1)
    return (images, labels)

def load_model() -> 'MockModel':
    if False:
        while True:
            i = 10
    'Load MNIST model.\n\n    Returns\n    -------\n    MnistModel\n    '
    path = MODEL_PATH
    saved_path = MODEL_SAVED_PATH

    def create_model():
        if False:
            return 10
        'Create a new model.'
        return keras.models.Sequential([keras.layers.Flatten(input_shape=(28, 28, 1)), keras.layers.Dense(128, activation='relu'), keras.layers.Dropout(0.2), keras.layers.Dense(10)])

    def add_softmax(model: keras.models.Sequential):
        if False:
            while True:
                i = 10
        'Add softmax layer to model.'
        return keras.Sequential([model, keras.layers.Softmax()])
    if saved_path.exists():
        model = create_model()
        model.load_weights(path).expect_partial()
        model = add_softmax(model)
        return MockModel(model)
    model = create_model()
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    (x_train, y_train) = load_mnist_data(train=True)
    model.fit(x_train, y_train, epochs=2)
    del x_train, y_train
    model.save_weights(path)
    model = add_softmax(model)
    if not path.parent.exists():
        path.parent.mkdir()
    return MockModel(model)

class MockModel:
    """Class of MNIST model that returns cached predictions."""

    def __init__(self, real_model):
        if False:
            return 10
        self.real_model = real_model
        with open(MNIST_DIR / 'static_predictions.pickle', 'rb') as handle:
            predictions = pickle.load(handle)
        self.cache = predictions

    def __call__(self, batch):
        if False:
            return 10
        results = []
        for img in batch:
            norm_img = (img - 0.1307) / 0.3081
            hash_key = hash_image(norm_img)
            if hash_key not in self.cache:
                prediction = self.real_model(np.expand_dims(img, 0)).numpy()
                self.cache[hash_key] = prediction[0]
            results.append(self.cache[hash_key])
        return np.stack(results)