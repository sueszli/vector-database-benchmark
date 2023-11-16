import copy
import json
import os
import random
import unittest
from pathlib import Path
from transformers.testing_utils import is_pipeline_test, require_decord, require_pytesseract, require_timm, require_torch, require_torch_or_tf, require_vision
from transformers.utils import direct_transformers_import, logging
from .pipelines.test_pipelines_audio_classification import AudioClassificationPipelineTests
from .pipelines.test_pipelines_automatic_speech_recognition import AutomaticSpeechRecognitionPipelineTests
from .pipelines.test_pipelines_conversational import ConversationalPipelineTests
from .pipelines.test_pipelines_depth_estimation import DepthEstimationPipelineTests
from .pipelines.test_pipelines_document_question_answering import DocumentQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_feature_extraction import FeatureExtractionPipelineTests
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
from .pipelines.test_pipelines_image_classification import ImageClassificationPipelineTests
from .pipelines.test_pipelines_image_segmentation import ImageSegmentationPipelineTests
from .pipelines.test_pipelines_image_to_image import ImageToImagePipelineTests
from .pipelines.test_pipelines_image_to_text import ImageToTextPipelineTests
from .pipelines.test_pipelines_mask_generation import MaskGenerationPipelineTests
from .pipelines.test_pipelines_object_detection import ObjectDetectionPipelineTests
from .pipelines.test_pipelines_question_answering import QAPipelineTests
from .pipelines.test_pipelines_summarization import SummarizationPipelineTests
from .pipelines.test_pipelines_table_question_answering import TQAPipelineTests
from .pipelines.test_pipelines_text2text_generation import Text2TextGenerationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests
from .pipelines.test_pipelines_text_to_audio import TextToAudioPipelineTests
from .pipelines.test_pipelines_token_classification import TokenClassificationPipelineTests
from .pipelines.test_pipelines_translation import TranslationPipelineTests
from .pipelines.test_pipelines_video_classification import VideoClassificationPipelineTests
from .pipelines.test_pipelines_visual_question_answering import VisualQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_zero_shot import ZeroShotClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_audio_classification import ZeroShotAudioClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_image_classification import ZeroShotImageClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_object_detection import ZeroShotObjectDetectionPipelineTests
pipeline_test_mapping = {'audio-classification': {'test': AudioClassificationPipelineTests}, 'automatic-speech-recognition': {'test': AutomaticSpeechRecognitionPipelineTests}, 'conversational': {'test': ConversationalPipelineTests}, 'depth-estimation': {'test': DepthEstimationPipelineTests}, 'document-question-answering': {'test': DocumentQuestionAnsweringPipelineTests}, 'feature-extraction': {'test': FeatureExtractionPipelineTests}, 'fill-mask': {'test': FillMaskPipelineTests}, 'image-classification': {'test': ImageClassificationPipelineTests}, 'image-segmentation': {'test': ImageSegmentationPipelineTests}, 'image-to-image': {'test': ImageToImagePipelineTests}, 'image-to-text': {'test': ImageToTextPipelineTests}, 'mask-generation': {'test': MaskGenerationPipelineTests}, 'object-detection': {'test': ObjectDetectionPipelineTests}, 'question-answering': {'test': QAPipelineTests}, 'summarization': {'test': SummarizationPipelineTests}, 'table-question-answering': {'test': TQAPipelineTests}, 'text2text-generation': {'test': Text2TextGenerationPipelineTests}, 'text-classification': {'test': TextClassificationPipelineTests}, 'text-generation': {'test': TextGenerationPipelineTests}, 'text-to-audio': {'test': TextToAudioPipelineTests}, 'token-classification': {'test': TokenClassificationPipelineTests}, 'translation': {'test': TranslationPipelineTests}, 'video-classification': {'test': VideoClassificationPipelineTests}, 'visual-question-answering': {'test': VisualQuestionAnsweringPipelineTests}, 'zero-shot': {'test': ZeroShotClassificationPipelineTests}, 'zero-shot-audio-classification': {'test': ZeroShotAudioClassificationPipelineTests}, 'zero-shot-image-classification': {'test': ZeroShotImageClassificationPipelineTests}, 'zero-shot-object-detection': {'test': ZeroShotObjectDetectionPipelineTests}}
for (task, task_info) in pipeline_test_mapping.items():
    test = task_info['test']
    task_info['mapping'] = {'pt': getattr(test, 'model_mapping', None), 'tf': getattr(test, 'tf_model_mapping', None)}
TRANSFORMERS_TINY_MODEL_PATH = os.environ.get('TRANSFORMERS_TINY_MODEL_PATH', 'hf-internal-testing')
if TRANSFORMERS_TINY_MODEL_PATH == 'hf-internal-testing':
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(Path(__file__).parent.parent, 'tests/utils/tiny_model_summary.json')
else:
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, 'reports', 'tiny_model_summary.json')
with open(TINY_MODEL_SUMMARY_FILE_PATH) as fp:
    tiny_model_summary = json.load(fp)
PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, 'src/transformers')
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)
logger = logging.get_logger(__name__)

class PipelineTesterMixin:
    model_tester = None
    pipeline_model_mapping = None
    supported_frameworks = ['pt', 'tf']

    def run_task_tests(self, task):
        if False:
            print('Hello World!')
        'Run pipeline tests for a specific `task`\n\n        Args:\n            task (`str`):\n                A task name. This should be a key in the mapping `pipeline_test_mapping`.\n        '
        if task not in self.pipeline_model_mapping:
            self.skipTest(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: `{task}` is not in `self.pipeline_model_mapping` for `{self.__class__.__name__}`.")
        model_architectures = self.pipeline_model_mapping[task]
        if not isinstance(model_architectures, tuple):
            model_architectures = (model_architectures,)
        if not isinstance(model_architectures, tuple):
            raise ValueError(f'`model_architectures` must be a tuple. Got {type(model_architectures)} instead.')
        for model_architecture in model_architectures:
            model_arch_name = model_architecture.__name__
            for _prefix in ['Flax', 'TF']:
                if model_arch_name.startswith(_prefix):
                    model_arch_name = model_arch_name[len(_prefix):]
                    break
            tokenizer_names = []
            processor_names = []
            commit = None
            if model_arch_name in tiny_model_summary:
                tokenizer_names = tiny_model_summary[model_arch_name]['tokenizer_classes']
                processor_names = tiny_model_summary[model_arch_name]['processor_classes']
                if 'sha' in tiny_model_summary[model_arch_name]:
                    commit = tiny_model_summary[model_arch_name]['sha']
            tokenizer_names = [None] if len(tokenizer_names) == 0 else tokenizer_names
            processor_names = [None] if len(processor_names) == 0 else processor_names
            repo_name = f'tiny-random-{model_arch_name}'
            if TRANSFORMERS_TINY_MODEL_PATH != 'hf-internal-testing':
                repo_name = model_arch_name
            self.run_model_pipeline_tests(task, repo_name, model_architecture, tokenizer_names, processor_names, commit)

    def run_model_pipeline_tests(self, task, repo_name, model_architecture, tokenizer_names, processor_names, commit):
        if False:
            for i in range(10):
                print('nop')
        'Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class names\n\n        Args:\n            task (`str`):\n                A task name. This should be a key in the mapping `pipeline_test_mapping`.\n            repo_name (`str`):\n                A model repository id on the Hub.\n            model_architecture (`type`):\n                A subclass of `PretrainedModel` or `PretrainedModel`.\n            tokenizer_names (`List[str]`):\n                A list of names of a subclasses of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.\n            processor_names (`List[str]`):\n                A list of names of subclasses of `BaseImageProcessor` or `FeatureExtractionMixin`.\n        '
        pipeline_test_class_name = pipeline_test_mapping[task]['test'].__name__
        for tokenizer_name in tokenizer_names:
            for processor_name in processor_names:
                if self.is_pipeline_test_to_skip(pipeline_test_class_name, model_architecture.config_class, model_architecture, tokenizer_name, processor_name):
                    logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: test is currently known to fail for: model `{model_architecture.__name__}` | tokenizer `{tokenizer_name}` | processor `{processor_name}`.")
                    continue
                self.run_pipeline_test(task, repo_name, model_architecture, tokenizer_name, processor_name, commit)

    def run_pipeline_test(self, task, repo_name, model_architecture, tokenizer_name, processor_name, commit):
        if False:
            return 10
        'Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class name\n\n        The model will be loaded from a model repository on the Hub.\n\n        Args:\n            task (`str`):\n                A task name. This should be a key in the mapping `pipeline_test_mapping`.\n            repo_name (`str`):\n                A model repository id on the Hub.\n            model_architecture (`type`):\n                A subclass of `PretrainedModel` or `PretrainedModel`.\n            tokenizer_name (`str`):\n                The name of a subclass of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.\n            processor_name (`str`):\n                The name of a subclass of `BaseImageProcessor` or `FeatureExtractionMixin`.\n        '
        repo_id = f'{TRANSFORMERS_TINY_MODEL_PATH}/{repo_name}'
        if TRANSFORMERS_TINY_MODEL_PATH != 'hf-internal-testing':
            model_type = model_architecture.config_class.model_type
            repo_id = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, model_type, repo_name)
        tokenizer = None
        if tokenizer_name is not None:
            tokenizer_class = getattr(transformers_module, tokenizer_name)
            tokenizer = tokenizer_class.from_pretrained(repo_id, revision=commit)
        processor = None
        if processor_name is not None:
            processor_class = getattr(transformers_module, processor_name)
            try:
                processor = processor_class.from_pretrained(repo_id, revision=commit)
            except Exception:
                logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not load the processor from `{repo_id}` with `{processor_name}`.")
                return
        if tokenizer is None and processor is None:
            logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load any tokenizer / processor from `{repo_id}`.")
            return
        try:
            model = model_architecture.from_pretrained(repo_id, revision=commit)
        except Exception:
            logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load the model from `{repo_id}` with `{model_architecture}`.")
            return
        pipeline_test_class_name = pipeline_test_mapping[task]['test'].__name__
        if self.is_pipeline_test_to_skip_more(pipeline_test_class_name, model.config, model, tokenizer, processor):
            logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: test is currently known to fail for: model `{model_architecture.__name__}` | tokenizer `{tokenizer_name}` | processor `{processor_name}`.")
            return
        validate_test_components(self, task, model, tokenizer, processor)
        if hasattr(model, 'eval'):
            model = model.eval()
        task_test = pipeline_test_mapping[task]['test']()
        (pipeline, examples) = task_test.get_test_pipeline(model, tokenizer, processor)
        if pipeline is None:
            logger.warning(f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not get the pipeline for testing.")
            return
        task_test.run_pipeline_test(pipeline, examples)

        def run_batch_test(pipeline, examples):
            if False:
                while True:
                    i = 10
            if pipeline.tokenizer is not None and pipeline.tokenizer.pad_token_id is None:
                return

            def data(n):
                if False:
                    i = 10
                    return i + 15
                for _ in range(n):
                    yield copy.deepcopy(random.choice(examples))
            out = []
            if task == 'conversational':
                for item in pipeline(data(10), batch_size=4, max_new_tokens=5):
                    out.append(item)
            else:
                for item in pipeline(data(10), batch_size=4):
                    out.append(item)
            self.assertEqual(len(out), 10)
        run_batch_test(pipeline, examples)

    @is_pipeline_test
    def test_pipeline_audio_classification(self):
        if False:
            print('Hello World!')
        self.run_task_tests(task='audio-classification')

    @is_pipeline_test
    def test_pipeline_automatic_speech_recognition(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='automatic-speech-recognition')

    @is_pipeline_test
    def test_pipeline_conversational(self):
        if False:
            print('Hello World!')
        self.run_task_tests(task='conversational')

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_depth_estimation(self):
        if False:
            print('Hello World!')
        self.run_task_tests(task='depth-estimation')

    @is_pipeline_test
    @require_pytesseract
    @require_torch
    @require_vision
    def test_pipeline_document_question_answering(self):
        if False:
            return 10
        self.run_task_tests(task='document-question-answering')

    @is_pipeline_test
    def test_pipeline_feature_extraction(self):
        if False:
            while True:
                i = 10
        self.run_task_tests(task='feature-extraction')

    @is_pipeline_test
    def test_pipeline_fill_mask(self):
        if False:
            while True:
                i = 10
        self.run_task_tests(task='fill-mask')

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    def test_pipeline_image_classification(self):
        if False:
            return 10
        self.run_task_tests(task='image-classification')

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_image_segmentation(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='image-segmentation')

    @is_pipeline_test
    @require_vision
    def test_pipeline_image_to_text(self):
        if False:
            return 10
        self.run_task_tests(task='image-to-text')

    @unittest.skip(reason='`run_pipeline_test` is currently not implemented.')
    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_mask_generation(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='mask-generation')

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_object_detection(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='object-detection')

    @is_pipeline_test
    def test_pipeline_question_answering(self):
        if False:
            return 10
        self.run_task_tests(task='question-answering')

    @is_pipeline_test
    def test_pipeline_summarization(self):
        if False:
            while True:
                i = 10
        self.run_task_tests(task='summarization')

    @is_pipeline_test
    def test_pipeline_table_question_answering(self):
        if False:
            print('Hello World!')
        self.run_task_tests(task='table-question-answering')

    @is_pipeline_test
    def test_pipeline_text2text_generation(self):
        if False:
            return 10
        self.run_task_tests(task='text2text-generation')

    @is_pipeline_test
    def test_pipeline_text_classification(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='text-classification')

    @is_pipeline_test
    @require_torch_or_tf
    def test_pipeline_text_generation(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='text-generation')

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_to_audio(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_task_tests(task='text-to-audio')

    @is_pipeline_test
    def test_pipeline_token_classification(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='token-classification')

    @is_pipeline_test
    def test_pipeline_translation(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_task_tests(task='translation')

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    @require_decord
    def test_pipeline_video_classification(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_task_tests(task='video-classification')

    @is_pipeline_test
    @require_torch
    @require_vision
    def test_pipeline_visual_question_answering(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_task_tests(task='visual-question-answering')

    @is_pipeline_test
    def test_pipeline_zero_shot(self):
        if False:
            return 10
        self.run_task_tests(task='zero-shot')

    @is_pipeline_test
    @require_torch
    def test_pipeline_zero_shot_audio_classification(self):
        if False:
            while True:
                i = 10
        self.run_task_tests(task='zero-shot-audio-classification')

    @is_pipeline_test
    @require_vision
    def test_pipeline_zero_shot_image_classification(self):
        if False:
            return 10
        self.run_task_tests(task='zero-shot-image-classification')

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_zero_shot_object_detection(self):
        if False:
            i = 10
            return i + 15
        self.run_task_tests(task='zero-shot-object-detection')

    def is_pipeline_test_to_skip(self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
        if False:
            while True:
                i = 10
        'Skip some tests based on the classes or their names without the instantiated objects.\n\n        This is to avoid calling `from_pretrained` (so reducing the runtime) if we already know the tests will fail.\n        '
        if pipeline_test_casse_name == 'DocumentQuestionAnsweringPipelineTests' and tokenizer_name is not None and (not tokenizer_name.endswith('Fast')):
            return True
        return False

    def is_pipeline_test_to_skip_more(self, pipeline_test_casse_name, config, model, tokenizer, processor):
        if False:
            return 10
        'Skip some more tests based on the information from the instantiated objects.'
        if pipeline_test_casse_name == 'QAPipelineTests' and tokenizer is not None and (getattr(tokenizer, 'pad_token', None) is None) and (not tokenizer.__class__.__name__.endswith('Fast')):
            return True
        return False

def validate_test_components(test_case, task, model, tokenizer, processor):
    if False:
        while True:
            i = 10
    if model.__class__.__name__ == 'BlenderbotForCausalLM':
        model.config.encoder_no_repeat_ngram_size = 0
    CONFIG_WITHOUT_VOCAB_SIZE = ['CanineConfig']
    if tokenizer is not None:
        config_vocab_size = getattr(model.config, 'vocab_size', None)
        if config_vocab_size is None:
            if hasattr(model.config, 'text_config'):
                config_vocab_size = getattr(model.config.text_config, 'vocab_size', None)
            elif hasattr(model.config, 'text_encoder'):
                config_vocab_size = getattr(model.config.text_encoder, 'vocab_size', None)
        if config_vocab_size is None and model.config.__class__.__name__ not in CONFIG_WITHOUT_VOCAB_SIZE:
            raise ValueError('Could not determine `vocab_size` from model configuration while `tokenizer` is not `None`.')