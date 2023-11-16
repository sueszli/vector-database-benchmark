import re
import numpy as np
import pandas as pd
import pytest
import torch
from ray.air.constants import MAX_REPR_LENGTH
from ray.air.util.data_batch_conversion import _convert_batch_type_to_pandas, _convert_pandas_to_batch_type
from ray.train.predictor import TYPE_TO_ENUM
from ray.train.tests.dummy_preprocessor import DummyPreprocessor
from ray.train.torch import TorchCheckpoint, TorchPredictor

class DummyModelSingleTensor(torch.nn.Module):

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        return input * 2

class DummyModelMultiInput(torch.nn.Module):

    def forward(self, input_dict):
        if False:
            for i in range(10):
                print('nop')
        return sum(input_dict.values())

class DummyModelMultiOutput(torch.nn.Module):

    def forward(self, input_tensor):
        if False:
            print('Hello World!')
        return {'a': input_tensor, 'b': input_tensor}

class DummyCustomModel(torch.nn.Module):
    """A model with an unsupported output type."""

    def forward(self, input_tensor):
        if False:
            for i in range(10):
                print('nop')
        return [input_tensor, input_tensor]

@pytest.fixture
def model():
    if False:
        i = 10
        return i + 15
    return DummyModelSingleTensor()

@pytest.fixture
def preprocessor():
    if False:
        i = 10
        return i + 15
    return DummyPreprocessor()

def test_repr(model):
    if False:
        print('Hello World!')
    predictor = TorchPredictor(model=model)
    representation = repr(predictor)
    assert len(representation) < MAX_REPR_LENGTH
    pattern = re.compile('^TorchPredictor\\((.*)\\)$')
    assert pattern.match(representation)

def test_init(model, preprocessor):
    if False:
        i = 10
        return i + 15
    predictor = TorchPredictor(model=model, preprocessor=preprocessor)
    checkpoint_predictor = TorchPredictor.from_checkpoint(TorchCheckpoint.from_model(model, preprocessor=preprocessor))
    data_batch = np.array([1, 2, 3])
    np.testing.assert_array_equal(predictor.predict(data_batch)['predictions'], checkpoint_predictor.predict(data_batch)['predictions'])
    assert checkpoint_predictor.get_preprocessor() == predictor.get_preprocessor()

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_model_not_training(model, use_gpu):
    if False:
        for i in range(10):
            print('nop')
    predictor = TorchPredictor(model=model, use_gpu=use_gpu)
    data_batch = np.array([1])
    predictor.predict(data_batch)
    assert not predictor.model.training

@pytest.mark.parametrize('batch_type', [np.ndarray, pd.DataFrame, dict])
def test_predict(batch_type):
    if False:
        for i in range(10):
            print('nop')
    predictor = TorchPredictor(model=DummyModelMultiInput())
    raw_batch = pd.DataFrame({'X0': [0.0, 0.0, 0.0], 'X1': [1.0, 2.0, 3.0]})
    data_batch = _convert_pandas_to_batch_type(raw_batch, type=TYPE_TO_ENUM[batch_type])
    raw_predictions = predictor.predict(data_batch, dtype=torch.float)
    predictions = _convert_batch_type_to_pandas(raw_predictions)
    assert len(predictions) == 3
    assert predictions.to_numpy().flatten().tolist() == [1.0, 2.0, 3.0]

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_array(model, use_gpu):
    if False:
        return 10
    predictor = TorchPredictor(model=model, use_gpu=use_gpu)
    data_batch = np.asarray([1, 2, 3])
    predictions = predictor.predict(data_batch)
    assert len(predictions) == 1
    np.testing.assert_array_equal(predictions['predictions'], np.asarray([2, 4, 6]))

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_array_with_preprocessor(model, preprocessor, use_gpu):
    if False:
        return 10
    predictor = TorchPredictor(model=model, preprocessor=preprocessor, use_gpu=use_gpu)
    data_batch = np.array([1, 2, 3])
    predictions = predictor.predict(data_batch)
    assert len(predictions) == 1
    np.testing.assert_array_equal(predictions['predictions'], np.asarray([2, 4, 6]))
    assert predictor.get_preprocessor().has_preprocessed

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_dataframe(use_gpu):
    if False:
        return 10
    predictor = TorchPredictor(model=DummyModelMultiInput(), use_gpu=use_gpu)
    data_batch = pd.DataFrame({'X0': [0.0, 0.0, 0.0], 'X1': [1.0, 2.0, 3.0]})
    predictions = predictor.predict(data_batch, dtype=torch.float)
    assert len(predictions) == 3
    assert predictions.to_numpy().flatten().tolist() == [1.0, 2.0, 3.0]

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_multi_output(use_gpu):
    if False:
        for i in range(10):
            print('nop')
    predictor = TorchPredictor(model=DummyModelMultiOutput(), use_gpu=use_gpu)
    data_batch = np.array([1, 2, 3])
    predictions = predictor.predict(data_batch)
    assert len(predictions) == 2
    for (k, v) in predictions.items():
        assert len(v) == 3
        assert v.flatten().tolist() == [1, 2, 3]

def test_predict_unsupported_output():
    if False:
        for i in range(10):
            print('nop')
    'Tests predictions with models that have unsupported output types.'
    predictor = TorchPredictor(model=DummyCustomModel())
    data_batch = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        predictor.predict(data_batch)

    class CustomPredictor(TorchPredictor):

        def call_model(self, tensor):
            if False:
                print('Hello World!')
            model_output = super().call_model(tensor)
            return {str(i): model_output[i] for i in range(len(model_output))}
    predictor = CustomPredictor(model=DummyCustomModel())
    predictions = predictor.predict(data_batch)
    assert len(predictions) == 2
    for (k, v) in predictions.items():
        assert len(v) == 3
        assert v.flatten().tolist() == [1, 2, 3]

@pytest.mark.parametrize('use_gpu', [False, True])
@pytest.mark.parametrize(('input_dtype', 'expected_output_dtype'), ((torch.float16, np.float16), (torch.float64, np.float64), (torch.int32, np.int32), (torch.int64, np.int64)))
def test_predict_array_with_different_dtypes(model, input_dtype, expected_output_dtype, use_gpu):
    if False:
        while True:
            i = 10
    predictor = TorchPredictor(model=model, use_gpu=use_gpu)
    data_batch = np.array([1, 2, 3])
    predictions = predictor.predict(data_batch, dtype=input_dtype)
    assert predictions['predictions'].dtype == expected_output_dtype

@pytest.mark.parametrize('use_gpu', [False, True])
def test_predict_array_no_training(model, use_gpu):
    if False:
        print('Hello World!')
    checkpoint = TorchCheckpoint.from_model(model)
    predictor = TorchPredictor.from_checkpoint(checkpoint, use_gpu=use_gpu)
    data_batch = np.array([1, 2, 3])
    predictions = predictor.predict(data_batch)
    assert len(predictions) == 1
    np.testing.assert_array_equal(predictions['predictions'], np.asarray([2, 4, 6]))

@pytest.mark.parametrize('use_gpu', [False, True])
def test_array_real_model(use_gpu):
    if False:
        i = 10
        return i + 15
    model = torch.nn.Linear(2, 1)
    predictor = TorchPredictor(model=model, use_gpu=use_gpu)
    data = np.array([[1, 2], [3, 4]])
    predictions = predictor.predict(data, dtype=torch.float)
    assert len(predictions) == 1
    assert len(predictions['predictions']) == 2

@pytest.mark.parametrize('use_gpu', [False, True])
def test_multi_modal_real_model(use_gpu):
    if False:
        print('Hello World!')

    class CustomModule(torch.nn.Module):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.linear1 = torch.nn.Linear(1, 1)
            self.linear2 = torch.nn.Linear(1, 1)

        def forward(self, input_dict: dict):
            if False:
                for i in range(10):
                    print('nop')
            input_dict['A'] = input_dict['A'].unsqueeze(1)
            input_dict['B'] = input_dict['B'].unsqueeze(1)
            out1 = self.linear1(input_dict['A'])
            out2 = self.linear2(input_dict['B'])
            return out1 + out2
    predictor = TorchPredictor(model=CustomModule(), use_gpu=use_gpu)
    data = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    predictions = predictor.predict(data, dtype=torch.float)
    assert len(predictions) == 2
    if use_gpu:
        assert next(predictor.model.parameters()).is_cuda, 'Model should be moved to GPU if use_gpu is True'
    else:
        assert not next(predictor.model.parameters()).is_cuda, 'Model should not be on GPU if use_gpu is False'
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', __file__]))