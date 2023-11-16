from unittest.mock import patch, Mock
import pytest
import torch
from haystack.preview.components.generators.hugging_face_local import HuggingFaceLocalGenerator, StopWordsCriteria

class TestHuggingFaceLocalGenerator:

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.model_info')
    def test_init_default(self, model_info_mock):
        if False:
            for i in range(10):
                print('nop')
        model_info_mock.return_value.pipeline_tag = 'text2text-generation'
        generator = HuggingFaceLocalGenerator()
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None}
        assert generator.generation_kwargs == {}
        assert generator.pipeline is None

    @pytest.mark.unit
    def test_init_custom_token(self):
        if False:
            i = 10
            return i + 15
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', token='test-token')
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': 'test-token'}

    @pytest.mark.unit
    def test_init_custom_device(self):
        if False:
            return 10
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', device='cuda:0')
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None, 'device': 'cuda:0'}

    @pytest.mark.unit
    def test_init_task_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        generator = HuggingFaceLocalGenerator(task='text2text-generation')
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None}

    @pytest.mark.unit
    def test_init_task_in_pipeline_kwargs(self):
        if False:
            i = 10
            return i + 15
        generator = HuggingFaceLocalGenerator(pipeline_kwargs={'task': 'text2text-generation'})
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None}

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.model_info')
    def test_init_task_inferred_from_model_name(self, model_info_mock):
        if False:
            for i in range(10):
                print('nop')
        model_info_mock.return_value.pipeline_tag = 'text2text-generation'
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base')
        assert generator.pipeline_kwargs == {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None}

    @pytest.mark.unit
    def test_init_invalid_task(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='is not supported.'):
            HuggingFaceLocalGenerator(task='text-classification')

    @pytest.mark.unit
    def test_init_pipeline_kwargs_override_other_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        pipeline_kwargs represent the main configuration of this component.\n        If they are provided, they should override other init parameters.\n        '
        pipeline_kwargs = {'model': 'gpt2', 'task': 'text-generation', 'device': 'cuda:0', 'token': 'another-test-token'}
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', device='cpu', token='test-token', pipeline_kwargs=pipeline_kwargs)
        assert generator.pipeline_kwargs == pipeline_kwargs

    @pytest.mark.unit
    def test_init_generation_kwargs(self):
        if False:
            print('Hello World!')
        generator = HuggingFaceLocalGenerator(task='text2text-generation', generation_kwargs={'max_new_tokens': 100})
        assert generator.generation_kwargs == {'max_new_tokens': 100}

    @pytest.mark.unit
    def test_init_set_return_full_text(self):
        if False:
            print('Hello World!')
        '\n        if not specified, return_full_text is set to False for text-generation task\n        (only generated text is returned, excluding prompt)\n        '
        generator = HuggingFaceLocalGenerator(task='text-generation')
        assert generator.generation_kwargs == {'return_full_text': False}

    @pytest.mark.unit
    def test_init_fails_with_both_stopwords_and_stoppingcriteria(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='Found both the `stop_words` init parameter and the `stopping_criteria` key in `generation_kwargs`'):
            HuggingFaceLocalGenerator(task='text2text-generation', stop_words=['coca', 'cola'], generation_kwargs={'stopping_criteria': 'fake-stopping-criteria'})

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.model_info')
    def test_to_dict_default(self, model_info_mock):
        if False:
            print('Hello World!')
        model_info_mock.return_value.pipeline_tag = 'text2text-generation'
        component = HuggingFaceLocalGenerator()
        data = component.to_dict()
        assert data == {'type': 'HuggingFaceLocalGenerator', 'init_parameters': {'pipeline_kwargs': {'model': 'google/flan-t5-base', 'task': 'text2text-generation', 'token': None}, 'generation_kwargs': {}, 'stop_words': None}}

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        component = HuggingFaceLocalGenerator(model_name_or_path='gpt2', task='text-generation', device='cuda:0', token='test-token', generation_kwargs={'max_new_tokens': 100}, stop_words=['coca', 'cola'])
        data = component.to_dict()
        assert data == {'type': 'HuggingFaceLocalGenerator', 'init_parameters': {'pipeline_kwargs': {'model': 'gpt2', 'task': 'text-generation', 'token': None, 'device': 'cuda:0'}, 'generation_kwargs': {'max_new_tokens': 100, 'return_full_text': False}, 'stop_words': ['coca', 'cola']}}

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.pipeline')
    def test_warm_up(self, pipeline_mock):
        if False:
            for i in range(10):
                print('nop')
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', token='test-token')
        pipeline_mock.assert_not_called()
        generator.warm_up()
        pipeline_mock.assert_called_once_with(model='google/flan-t5-base', task='text2text-generation', token='test-token')

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.pipeline')
    def test_warm_up_doesn_reload(self, pipeline_mock):
        if False:
            for i in range(10):
                print('nop')
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', token='test-token')
        pipeline_mock.assert_not_called()
        generator.warm_up()
        generator.warm_up()
        pipeline_mock.assert_called_once()

    @pytest.mark.unit
    def test_run(self):
        if False:
            return 10
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', generation_kwargs={'max_new_tokens': 100})
        generator.pipeline = Mock(return_value=[{'generated_text': 'Rome'}])
        results = generator.run(prompt="What's the capital of Italy?")
        generator.pipeline.assert_called_once_with("What's the capital of Italy?", max_new_tokens=100, stopping_criteria=None)
        assert results == {'replies': ['Rome']}

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.pipeline')
    def test_run_empty_prompt(self, pipeline_mock):
        if False:
            return 10
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', generation_kwargs={'max_new_tokens': 100})
        generator.warm_up()
        results = generator.run(prompt='')
        assert results == {'replies': []}

    @pytest.mark.unit
    def test_run_with_generation_kwargs(self):
        if False:
            while True:
                i = 10
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', generation_kwargs={'max_new_tokens': 100})
        generator.pipeline = Mock(return_value=[{'generated_text': 'Rome'}])
        generator.run(prompt='irrelevant', generation_kwargs={'max_new_tokens': 200, 'temperature': 0.5})
        generator.pipeline.assert_called_once_with('irrelevant', max_new_tokens=200, temperature=0.5, stopping_criteria=None)

    @pytest.mark.unit
    def test_run_fails_without_warm_up(self):
        if False:
            i = 10
            return i + 15
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', generation_kwargs={'max_new_tokens': 100})
        with pytest.raises(RuntimeError, match='The generation model has not been loaded.'):
            generator.run(prompt='irrelevant')

    @pytest.mark.unit
    def test_stop_words_criteria(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that StopWordsCriteria will check stop word tokens in a continuous and sequential order\n        '
        stop_words_id = torch.tensor([[73, 24621, 11937]])
        input_ids1 = torch.tensor([[100, 19, 24621, 11937, 6, 68, 19, 73, 3897, 5]])
        input_ids2 = torch.tensor([[100, 19, 73, 24621, 11937]])
        stop_words_criteria = StopWordsCriteria(tokenizer=Mock(), stop_words=['mock data'])
        stop_words_criteria.stop_ids = stop_words_id
        present_and_continuous = stop_words_criteria(input_ids1, scores=None)
        assert not present_and_continuous
        present_and_continuous = stop_words_criteria(input_ids2, scores=None)
        assert present_and_continuous

    @pytest.mark.unit
    @patch('haystack.preview.components.generators.hugging_face_local.pipeline')
    @patch('haystack.preview.components.generators.hugging_face_local.StopWordsCriteria')
    @patch('haystack.preview.components.generators.hugging_face_local.StoppingCriteriaList')
    def test_warm_up_set_stopping_criteria_list(self, pipeline_mock, stop_words_criteria_mock, stopping_criteria_list_mock):
        if False:
            return 10
        '\n        Test that warm_up method sets the `stopping_criteria_list` attribute\n        if `stop_words` is provided\n        '
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', stop_words=['coca', 'cola'])
        generator.warm_up()
        stop_words_criteria_mock.assert_called_once()
        stopping_criteria_list_mock.assert_called_once()
        assert hasattr(generator, 'stopping_criteria_list')

    @pytest.mark.unit
    def test_run_stop_words_removal(self):
        if False:
            while True:
                i = 10
        '\n        Test that stop words are removed from the generated text\n        (does not test stopping text generation)\n        '
        generator = HuggingFaceLocalGenerator(model_name_or_path='google/flan-t5-base', task='text2text-generation', stop_words=['world'])
        generator.pipeline = Mock(return_value=[{'generated_text': 'Hello world'}])
        results = generator.run(prompt='irrelevant')
        assert results == {'replies': ['Hello']}