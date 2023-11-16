import copy
import json
from os import PathLike
import random
from typing import Any, Dict, Iterable, Set, Union
import torch
import numpy
from numpy.testing import assert_allclose
from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.data.batch import Batch
from allennlp.models import load_archive, Model
from allennlp.training import GradientDescentTrainer
from allennlp.confidence_checks.normalization_bias_verification import NormalizationBiasVerification

class ModelTestCase(AllenNlpTestCase):
    """
    A subclass of [`AllenNlpTestCase`](./test_case.md)
    with added methods for testing [`Model`](../../models/model.md) subclasses.
    """

    def set_up_model(self, param_file: PathLike, dataset_file: PathLike, serialization_dir: PathLike=None, seed: int=None):
        if False:
            i = 10
            return i + 15
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        self.param_file = str(param_file)
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'], serialization_dir=serialization_dir)
        instances = list(reader.read(str(dataset_file)))
        if 'vocabulary' in params:
            vocab_params = params['vocabulary']
            vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
        else:
            vocab = Vocabulary.from_instances(instances)
        self.vocab = vocab
        self.instances = instances
        self.model = Model.from_params(vocab=self.vocab, params=params['model'], serialization_dir=serialization_dir)
        self.dataset = Batch(self.instances)
        self.dataset.index_instances(self.vocab)

    def test_model_batch_norm_verification(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'model'):
            verification = NormalizationBiasVerification(self.model)
            assert verification.check(inputs=self.dataset.as_tensor_dict())

    def ensure_model_can_train_save_and_load(self, param_file: Union[PathLike, str], tolerance: float=0.0001, cuda_device: int=-1, gradients_to_ignore: Set[str]=None, overrides: str='', metric_to_check: str=None, metric_terminal_value: float=None, metric_tolerance: float=0.0001, disable_dropout: bool=True, which_loss: str='loss', seed: int=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        # Parameters\n\n        param_file : `str`\n            Path to a training configuration file that we will use to train the model for this\n            test.\n        tolerance : `float`, optional (default=`1e-4`)\n            When comparing model predictions between the originally-trained model and the model\n            after saving and loading, we will use this tolerance value (passed as `rtol` to\n            `numpy.testing.assert_allclose`).\n        cuda_device : `int`, optional (default=`-1`)\n            The device to run the test on.\n        gradients_to_ignore : `Set[str]`, optional (default=`None`)\n            This test runs a gradient check to make sure that we\'re actually computing gradients\n            for all of the parameters in the model.  If you really want to ignore certain\n            parameters when doing that check, you can pass their names here.  This is not\n            recommended unless you\'re `really` sure you don\'t need to have non-zero gradients for\n            those parameters (e.g., some of the beam search / state machine models have\n            infrequently-used parameters that are hard to force the model to use in a small test).\n        overrides : `str`, optional (default = `""`)\n            A JSON string that we will use to override values in the input parameter file.\n        metric_to_check: `str`, optional (default = `None`)\n            We may want to automatically perform a check that model reaches given metric when\n            training (on validation set, if it is specified). It may be useful in CI, for example.\n            You can pass any metric that is in your model returned metrics.\n        metric_terminal_value: `str`, optional (default = `None`)\n            When you set `metric_to_check`, you need to set the value this metric must converge to\n        metric_tolerance: `float`, optional (default=`1e-4`)\n            Tolerance to check you model metric against metric terminal value. One can expect some\n            variance in model metrics when the training process is highly stochastic.\n        disable_dropout : `bool`, optional (default = `True`)\n            If True we will set all dropout to 0 before checking gradients. (Otherwise, with small\n            datasets, you may get zero gradients because of unlucky dropout.)\n        which_loss: `str`, optional (default = `"loss"`)\n            Specifies which loss to test. For example, which_loss may be "adversary_loss" for\n            `adversarial_bias_mitigator`.\n        '
        save_dir = self.TEST_DIR / 'save_and_load_test'
        archive_file = save_dir / 'model.tar.gz'
        model = train_model_from_file(param_file, save_dir, overrides=overrides, return_model=True)
        assert model is not None
        metrics_file = save_dir / 'metrics.json'
        if metric_to_check is not None:
            metrics = json.loads(metrics_file.read_text())
            metric_value = metrics.get(f'best_validation_{metric_to_check}') or metrics.get(f'training_{metric_to_check}')
            assert metric_value is not None, f'Cannot find {metric_to_check} in metrics.json file'
            assert metric_terminal_value is not None, 'Please specify metric terminal value'
            assert abs(metric_value - metric_terminal_value) < metric_tolerance
        archive = load_archive(archive_file, cuda_device=cuda_device)
        loaded_model = archive.model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        for key in state_keys:
            assert_allclose(model.state_dict()[key].cpu().numpy(), loaded_model.state_dict()[key].cpu().numpy(), err_msg=key)
        reader = archive.dataset_reader
        params = Params.from_file(param_file, params_overrides=overrides)
        data_loader_params = params['data_loader']
        data_loader_params['shuffle'] = False
        data_loader_params2 = Params(copy.deepcopy(data_loader_params.as_dict()))
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        print('Reading with original model')
        data_loader = DataLoader.from_params(params=data_loader_params, reader=reader, data_path=params['validation_data_path'])
        data_loader.index_with(model.vocab)
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        print('Reading with loaded model')
        data_loader2 = DataLoader.from_params(params=data_loader_params2, reader=reader, data_path=params['validation_data_path'])
        data_loader2.index_with(loaded_model.vocab)
        model_batch = next(iter(data_loader))
        loaded_batch = next(iter(data_loader2))
        self.check_model_computes_gradients_correctly(model, model_batch, gradients_to_ignore, disable_dropout, which_loss)
        assert model_batch.keys() == loaded_batch.keys()
        for key in model_batch.keys():
            self.assert_fields_equal(model_batch[key], loaded_batch[key], key, 1e-06)
        model.eval()
        loaded_model.eval()
        for model_ in [model, loaded_model]:
            for module in model_.modules():
                if hasattr(module, 'stateful') and module.stateful:
                    module.reset_states()
        print('Predicting with original model')
        model_predictions = model(**model_batch)
        print('Predicting with loaded model')
        loaded_model_predictions = loaded_model(**loaded_batch)
        for key in model_predictions.keys():
            self.assert_fields_equal(model_predictions[key], loaded_model_predictions[key], name=key, tolerance=tolerance)
        loaded_model.train()
        loaded_model_predictions = loaded_model(**loaded_batch)
        loaded_model_loss = loaded_model_predictions[which_loss]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()
        return (model, loaded_model)

    def ensure_model_can_train(self, trainer: GradientDescentTrainer, gradients_to_ignore: Set[str]=None, metric_to_check: str=None, metric_terminal_value: float=None, metric_tolerance: float=0.0001, disable_dropout: bool=True):
        if False:
            print('Hello World!')
        "\n        A simple test for model training behavior when you are not using configuration files. In\n        this case, we don't have a story around saving and loading models (you need to handle that\n        yourself), so we don't have tests for that.  We just test that the model can train, and that\n        it computes gradients for all parameters.\n\n        Because the `Trainer` already has a reference to a model and to a data loader, we just take\n        the `Trainer` object itself, and grab the `Model` and other necessary objects from there.\n\n        # Parameters\n\n        trainer: `GradientDescentTrainer`\n            The `Trainer` to use for the test, which already has references to a `Model` and a\n            `DataLoader`, which we will use in the test.\n        gradients_to_ignore : `Set[str]`, optional (default=`None`)\n            This test runs a gradient check to make sure that we're actually computing gradients\n            for all of the parameters in the model.  If you really want to ignore certain\n            parameters when doing that check, you can pass their names here.  This is not\n            recommended unless you're `really` sure you don't need to have non-zero gradients for\n            those parameters (e.g., some of the beam search / state machine models have\n            infrequently-used parameters that are hard to force the model to use in a small test).\n        metric_to_check: `str`, optional (default = `None`)\n            We may want to automatically perform a check that model reaches given metric when\n            training (on validation set, if it is specified). It may be useful in CI, for example.\n            You can pass any metric that is in your model returned metrics.\n        metric_terminal_value: `str`, optional (default = `None`)\n            When you set `metric_to_check`, you need to set the value this metric must converge to\n        metric_tolerance: `float`, optional (default=`1e-4`)\n            Tolerance to check you model metric against metric terminal value. One can expect some\n            variance in model metrics when the training process is highly stochastic.\n        disable_dropout : `bool`, optional (default = `True`)\n            If True we will set all dropout to 0 before checking gradients. (Otherwise, with small\n            datasets, you may get zero gradients because of unlucky dropout.)\n        "
        metrics = trainer.train()
        if metric_to_check is not None:
            metric_value = metrics.get(f'best_validation_{metric_to_check}') or metrics.get(f'training_{metric_to_check}')
            assert metric_value is not None, f'Cannot find {metric_to_check} in metrics.json file'
            assert metric_terminal_value is not None, 'Please specify metric terminal value'
            assert abs(metric_value - metric_terminal_value) < metric_tolerance
        model_batch = next(iter(trainer.data_loader))
        self.check_model_computes_gradients_correctly(trainer.model, model_batch, gradients_to_ignore, disable_dropout)

    def assert_fields_equal(self, field1, field2, name: str, tolerance: float=1e-06) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(field1, torch.Tensor):
            assert_allclose(field1.detach().cpu().numpy(), field2.detach().cpu().numpy(), rtol=tolerance, err_msg=name)
        elif isinstance(field1, dict):
            assert field1.keys() == field2.keys()
            for key in field1:
                self.assert_fields_equal(field1[key], field2[key], tolerance=tolerance, name=name + '.' + str(key))
        elif isinstance(field1, (list, tuple)):
            assert len(field1) == len(field2)
            for (i, (subfield1, subfield2)) in enumerate(zip(field1, field2)):
                self.assert_fields_equal(subfield1, subfield2, tolerance=tolerance, name=name + f'[{i}]')
        elif isinstance(field1, (float, int)):
            assert_allclose([field1], [field2], rtol=tolerance, err_msg=name)
        else:
            if field1 != field2:
                for key in field1.__dict__:
                    print(key, getattr(field1, key) == getattr(field2, key))
            assert field1 == field2, f'{name}, {type(field1)}, {type(field2)}'

    @staticmethod
    def check_model_computes_gradients_correctly(model: Model, model_batch: Dict[str, Union[Any, Dict[str, Any]]], params_to_ignore: Set[str]=None, disable_dropout: bool=True, which_loss: str='loss'):
        if False:
            for i in range(10):
                print('nop')
        print('Checking gradients')
        for p in model.parameters():
            p.grad = None
        model.train()
        original_dropouts: Dict[str, float] = {}
        if disable_dropout:
            for (name, module) in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    original_dropouts[name] = getattr(module, 'p')
                    setattr(module, 'p', 0)
        result = model(**model_batch)
        result[which_loss].backward()
        has_zero_or_none_grads = {}
        for (name, parameter) in model.named_parameters():
            zeros = torch.zeros(parameter.size())
            if params_to_ignore and name in params_to_ignore:
                continue
            if parameter.requires_grad:
                if parameter.grad is None:
                    has_zero_or_none_grads[name] = 'No gradient computed (i.e parameter.grad is None)'
                elif parameter.grad.is_sparse or parameter.grad.data.is_sparse:
                    pass
                elif (parameter.grad.cpu() == zeros).all():
                    has_zero_or_none_grads[name] = f'zeros with shape ({tuple(parameter.grad.size())})'
            else:
                assert parameter.grad is None
        if has_zero_or_none_grads:
            for (name, grad) in has_zero_or_none_grads.items():
                print(f'Parameter: {name} had incorrect gradient: {grad}')
            raise Exception('Incorrect gradients found. See stdout for more info.')
        if disable_dropout:
            for (name, module) in model.named_modules():
                if name in original_dropouts:
                    setattr(module, 'p', original_dropouts[name])

    def ensure_batch_predictions_are_consistent(self, keys_to_ignore: Iterable[str]=()):
        if False:
            return 10
        '\n        Ensures that the model performs the same on a batch of instances as on individual instances.\n        Ignores metrics matching the regexp .*loss.* and those specified explicitly.\n\n        # Parameters\n\n        keys_to_ignore : `Iterable[str]`, optional (default=`()`)\n            Names of metrics that should not be taken into account, e.g. "batch_weight".\n        '
        self.model.eval()
        single_predictions = []
        for (i, instance) in enumerate(self.instances):
            dataset = Batch([instance])
            tensors = dataset.as_tensor_dict(dataset.get_padding_lengths())
            result = self.model(**tensors)
            single_predictions.append(result)
        full_dataset = Batch(self.instances)
        batch_tensors = full_dataset.as_tensor_dict(full_dataset.get_padding_lengths())
        batch_predictions = self.model(**batch_tensors)
        for (i, instance_predictions) in enumerate(single_predictions):
            for (key, single_predicted) in instance_predictions.items():
                tolerance = 1e-06
                if 'loss' in key:
                    continue
                if key in keys_to_ignore:
                    continue
                single_predicted = single_predicted[0]
                batch_predicted = batch_predictions[key][i]
                if isinstance(single_predicted, torch.Tensor):
                    if single_predicted.size() != batch_predicted.size():
                        slices = tuple((slice(0, size) for size in single_predicted.size()))
                        batch_predicted = batch_predicted[slices]
                    assert_allclose(single_predicted.data.numpy(), batch_predicted.data.numpy(), atol=tolerance, err_msg=key)
                else:
                    assert single_predicted == batch_predicted, key