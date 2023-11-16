from flaky import flaky
import numpy
import pytest
import torch
from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Trainer

class SimpleTaggerTest(ModelTestCase):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        super().setup_method()
        self.set_up_model(self.FIXTURES_ROOT / 'simple_tagger' / 'experiment.json', self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')

    def test_simple_tagger_can_train_save_and_load(self):
        if False:
            return 10
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        if False:
            i = 10
            return i + 15
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        class_probs = output_dict['class_probabilities'][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1), numpy.array([1, 1, 1, 1]))

    def test_forward_on_instances_ignores_loss_key_when_batched(self):
        if False:
            i = 10
            return i + 15
        batch_outputs = self.model.forward_on_instances(self.dataset.instances)
        for output in batch_outputs:
            assert 'loss' not in output.keys()
        single_output = self.model.forward_on_instance(self.dataset.instances[0])
        assert 'loss' in single_output.keys()

    def test_mismatching_dimensions_throws_configuration_error(self):
        if False:
            while True:
                i = 10
        params = Params.from_file(self.param_file)
        params['model']['encoder']['input_size'] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop('model'))

    def test_regularization(self):
        if False:
            for i in range(10):
                print('nop')
        penalty = self.model.get_regularization_penalty()
        assert penalty is None
        data_loader = SimpleDataLoader(self.instances, batch_size=32)
        trainer = GradientDescentTrainer(self.model, None, data_loader)
        training_batch = next(iter(data_loader))
        validation_batch = next(iter(data_loader))
        training_loss = trainer.batch_outputs(training_batch, for_training=True)['loss'].item()
        validation_loss = trainer.batch_outputs(validation_batch, for_training=False)['loss'].item()
        numpy.testing.assert_almost_equal(training_loss, validation_loss)

class SimpleTaggerSpanF1Test(ModelTestCase):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.set_up_model(self.FIXTURES_ROOT / 'simple_tagger_with_span_f1' / 'experiment.json', self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_simple_tagger_can_train_save_and_load(self):
        if False:
            while True:
                i = 10
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        if False:
            return 10
        self.ensure_batch_predictions_are_consistent()

    def test_simple_tagger_can_enable_span_f1(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.model.calculate_span_f1 and self.model._f1_metric is not None

class SimpleTaggerRegularizationTest(ModelTestCase):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        super().setup_method()
        param_file = self.FIXTURES_ROOT / 'simple_tagger' / 'experiment_with_regularization.json'
        self.set_up_model(param_file, self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')
        params = Params.from_file(param_file)
        self.reader = DatasetReader.from_params(params['dataset_reader'])
        self.data_loader = DataLoader.from_params(reader=self.reader, data_path=str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'), params=params['data_loader'])
        self.data_loader.index_with(self.vocab)
        self.trainer = Trainer.from_params(model=self.model, data_loader=self.data_loader, serialization_dir=self.TEST_DIR, params=params.get('trainer'))

    def test_regularization(self):
        if False:
            i = 10
            return i + 15
        penalty = self.model.get_regularization_penalty().data
        assert (penalty > 0).all()
        penalty2 = 0
        for (name, parameter) in self.model.named_parameters():
            if name.endswith('weight'):
                weight_penalty = 10 * torch.sum(torch.pow(parameter, 2))
                penalty2 += weight_penalty
            elif name.endswith('bias'):
                bias_penalty = 5 * torch.sum(torch.abs(parameter))
                penalty2 += bias_penalty
        assert (penalty == penalty2.data).all()
        training_batch = next(iter(self.data_loader))
        validation_batch = next(iter(self.data_loader))
        training_batch_outputs = self.trainer.batch_outputs(training_batch, for_training=True)
        training_loss = training_batch_outputs['loss'].data
        assert (penalty == training_batch_outputs['reg_loss']).all()
        validation_loss = self.trainer.batch_outputs(validation_batch, for_training=False)['loss'].data
        assert (training_loss != validation_loss).all()
        penalized = validation_loss + penalty
        assert (training_loss == penalized).all()