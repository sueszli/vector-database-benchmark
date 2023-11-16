import argparse
import collections.abc
import copy
import inspect
import json
import multiprocessing
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from check_config_docstrings import get_checkpoint_from_config_class
from datasets import load_dataset
from get_test_info import get_model_to_tester_mapping, get_tester_classes_for_model
from huggingface_hub import Repository, create_repo, hf_api, upload_folder
from transformers import CONFIG_MAPPING, FEATURE_EXTRACTOR_MAPPING, IMAGE_PROCESSOR_MAPPING, PROCESSOR_MAPPING, TOKENIZER_MAPPING, AutoTokenizer, LayoutLMv3TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast, logging
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.configuration_auto import AutoConfig, model_type_to_module_name
from transformers.models.fsmt import configuration_fsmt
from transformers.processing_utils import ProcessorMixin, transformers_module
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.set_verbosity_error()
logging.disable_progress_bar()
logger = logging.get_logger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not is_torch_available():
    raise ValueError('Please install PyTorch.')
if not is_tf_available():
    raise ValueError('Please install TensorFlow.')
FRAMEWORKS = ['pytorch', 'tensorflow']
INVALID_ARCH = []
TARGET_VOCAB_SIZE = 1024
data = {'training_ds': None, 'testing_ds': None}
COMPOSITE_MODELS = {'EncoderDecoderModel': 'EncoderDecoderModel-bert-bert', 'SpeechEncoderDecoderModel': 'SpeechEncoderDecoderModel-wav2vec2-bert', 'VisionEncoderDecoderModel': 'VisionEncoderDecoderModel-vit-gpt2', 'VisionTextDualEncoderModel': 'VisionTextDualEncoderModel-vit-bert'}
UNCONVERTIBLE_MODEL_ARCHITECTURES = {'BertGenerationEncoder', 'BertGenerationDecoder', 'CamembertForSequenceClassification', 'CamembertForMultipleChoice', 'CamembertForMaskedLM', 'CamembertForCausalLM', 'CamembertForTokenClassification', 'CamembertForQuestionAnswering', 'CamembertModel', 'TFCamembertForMultipleChoice', 'TFCamembertForTokenClassification', 'TFCamembertForQuestionAnswering', 'TFCamembertForSequenceClassification', 'TFCamembertForMaskedLM', 'TFCamembertModel', 'TFCamembertForCausalLM', 'DecisionTransformerModel', 'GraphormerModel', 'InformerModel', 'JukeboxModel', 'MarianForCausalLM', 'MaskFormerSwinModel', 'MaskFormerSwinBackbone', 'MT5Model', 'MT5ForConditionalGeneration', 'UMT5ForConditionalGeneration', 'TFMT5ForConditionalGeneration', 'TFMT5Model', 'QDQBertForSequenceClassification', 'QDQBertForMaskedLM', 'QDQBertModel', 'QDQBertForTokenClassification', 'QDQBertLMHeadModel', 'QDQBertForMultipleChoice', 'QDQBertForQuestionAnswering', 'QDQBertForNextSentencePrediction', 'ReformerModelWithLMHead', 'RetriBertModel', 'Speech2Text2ForCausalLM', 'TimeSeriesTransformerModel', 'TrajectoryTransformerModel', 'TrOCRForCausalLM', 'XLMProphetNetForConditionalGeneration', 'XLMProphetNetForCausalLM', 'XLMProphetNetModel', 'XLMRobertaModel', 'XLMRobertaForTokenClassification', 'XLMRobertaForMultipleChoice', 'XLMRobertaForMaskedLM', 'XLMRobertaForCausalLM', 'XLMRobertaForSequenceClassification', 'XLMRobertaForQuestionAnswering', 'TFXLMRobertaForSequenceClassification', 'TFXLMRobertaForMaskedLM', 'TFXLMRobertaForCausalLM', 'TFXLMRobertaForQuestionAnswering', 'TFXLMRobertaModel', 'TFXLMRobertaForMultipleChoice', 'TFXLMRobertaForTokenClassification'}

def get_processor_types_from_config_class(config_class, allowed_mappings=None):
    if False:
        i = 10
        return i + 15
    'Return a tuple of processors for `config_class`.\n\n    We use `tuple` here to include (potentially) both slow & fast tokenizers.\n    '

    def _to_tuple(x):
        if False:
            return 10
        if not isinstance(x, collections.abc.Sequence):
            x = (x,)
        else:
            x = tuple(x)
        return x
    if allowed_mappings is None:
        allowed_mappings = ['processor', 'tokenizer', 'image_processor', 'feature_extractor']
    processor_types = ()
    if config_class in PROCESSOR_MAPPING and 'processor' in allowed_mappings:
        processor_types = _to_tuple(PROCESSOR_MAPPING[config_class])
    else:
        if config_class in TOKENIZER_MAPPING and 'tokenizer' in allowed_mappings:
            processor_types = TOKENIZER_MAPPING[config_class]
        if config_class in IMAGE_PROCESSOR_MAPPING and 'image_processor' in allowed_mappings:
            processor_types += _to_tuple(IMAGE_PROCESSOR_MAPPING[config_class])
        elif config_class in FEATURE_EXTRACTOR_MAPPING and 'feature_extractor' in allowed_mappings:
            processor_types += _to_tuple(FEATURE_EXTRACTOR_MAPPING[config_class])
    processor_types = tuple((p for p in processor_types if p is not None))
    return processor_types

def get_architectures_from_config_class(config_class, arch_mappings, models_to_skip=None):
    if False:
        return 10
    'Return a tuple of all possible architectures attributed to a configuration class `config_class`.\n\n    For example, BertConfig -> [BertModel, BertForMaskedLM, ..., BertForQuestionAnswering].\n    '
    architectures = set()
    if models_to_skip is None:
        models_to_skip = []
    models_to_skip = UNCONVERTIBLE_MODEL_ARCHITECTURES.union(models_to_skip)
    for mapping in arch_mappings:
        if config_class in mapping:
            models = mapping[config_class]
            models = tuple(models) if isinstance(models, collections.abc.Sequence) else (models,)
            for model in models:
                if model.__name__ not in models_to_skip:
                    architectures.add(model)
    architectures = tuple(architectures)
    return architectures

def get_config_class_from_processor_class(processor_class):
    if False:
        return 10
    'Get the config class from a processor class.\n\n    Some config/model classes use tokenizers/feature_extractors from other models. For example, `GPT-J` uses\n    `GPT2Tokenizer`. If no checkpoint is found for a config class, or a checkpoint is found without necessary file(s) to\n    create the processor for `processor_class`, we get the config class that corresponds to `processor_class` and use it\n    to find a checkpoint in order to create the processor.\n    '
    processor_prefix = processor_class.__name__
    for postfix in ['TokenizerFast', 'Tokenizer', 'ImageProcessor', 'FeatureExtractor', 'Processor']:
        processor_prefix = processor_prefix.replace(postfix, '')
    if processor_prefix == 'Wav2Vec2CTC':
        processor_prefix = 'Wav2Vec2'
    new_config_name = f'{processor_prefix}Config'
    new_config_class = getattr(transformers_module, new_config_name)
    return new_config_class

def build_processor(config_class, processor_class, allow_no_checkpoint=False):
    if False:
        print('Hello World!')
    'Create a processor for `processor_class`.\n\n    If a processor is not able to be built with the original arguments, this method tries to change the arguments and\n    call itself recursively, by inferring a new `config_class` or a new `processor_class` from another one, in order to\n    find a checkpoint containing the necessary files to build a processor.\n\n    The processor is not saved here. Instead, it will be saved in `convert_processors` after further changes in\n    `convert_processors`. For each model architecture`, a copy will be created and saved along the built model.\n    '
    checkpoint = get_checkpoint_from_config_class(config_class)
    if checkpoint is None:
        config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
        checkpoint = get_checkpoint_from_config_class(config_class_from_processor_class)
    processor = None
    try:
        processor = processor_class.from_pretrained(checkpoint)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}')
    if processor is None and checkpoint is not None and issubclass(processor_class, (PreTrainedTokenizerBase, AutoTokenizer)):
        try:
            config = AutoConfig.from_pretrained(checkpoint)
        except Exception as e:
            logger.error(f'{e.__class__.__name__}: {e}')
            config = None
        if config is not None:
            if not isinstance(config, config_class):
                raise ValueError(f'`config` (which is of type {config.__class__.__name__}) should be an instance of `config_class` ({config_class.__name__})!')
            tokenizer_class = config.tokenizer_class
            new_processor_class = None
            if tokenizer_class is not None:
                new_processor_class = getattr(transformers_module, tokenizer_class)
                if new_processor_class != processor_class:
                    processor = build_processor(config_class, new_processor_class)
            if processor is None:
                new_processor_classes = get_processor_types_from_config_class(config.__class__, allowed_mappings=['tokenizer'])
                names = [x.__name__.replace('Fast', '') for x in [processor_class, new_processor_class] if x is not None]
                new_processor_classes = [x for x in new_processor_classes if x is not None and x.__name__.replace('Fast', '') not in names]
                if len(new_processor_classes) > 0:
                    new_processor_class = new_processor_classes[0]
                    for x in new_processor_classes:
                        if x.__name__.endswith('Fast'):
                            new_processor_class = x
                            break
                    processor = build_processor(config_class, new_processor_class)
    if processor is None:
        if issubclass(processor_class, ProcessorMixin):
            attrs = {}
            for attr_name in processor_class.attributes:
                attrs[attr_name] = []
                attr_class_names = getattr(processor_class, f'{attr_name}_class')
                if not isinstance(attr_class_names, tuple):
                    attr_class_names = (attr_class_names,)
                for name in attr_class_names:
                    attr_class = getattr(transformers_module, name)
                    attr = build_processor(config_class, attr_class)
                    if attr is not None:
                        attrs[attr_name].append(attr)
            if all((len(v) > 0 for v in attrs.values())):
                try:
                    processor = processor_class(**{k: v[0] for (k, v) in attrs.items()})
                except Exception as e:
                    logger.error(f'{e.__class__.__name__}: {e}')
        else:
            config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
            if config_class_from_processor_class != config_class:
                processor = build_processor(config_class_from_processor_class, processor_class)
    if processor is None and allow_no_checkpoint and (issubclass(processor_class, BaseImageProcessor) or issubclass(processor_class, FeatureExtractionMixin)):
        try:
            processor = processor_class()
        except Exception as e:
            logger.error(f'{e.__class__.__name__}: {e}')
    if processor is not None:
        if not (isinstance(processor, processor_class) or processor_class.__name__.startswith('Auto')):
            raise ValueError(f'`processor` (which is of type {processor.__class__.__name__}) should be an instance of {processor_class.__name__} or an Auto class!')
    return processor

def get_tiny_config(config_class, model_class=None, **model_tester_kwargs):
    if False:
        while True:
            i = 10
    "Retrieve a tiny configuration from `config_class` using each model's `ModelTester`.\n\n    Args:\n        config_class: Subclass of `PreTrainedConfig`.\n\n    Returns:\n        An instance of `config_class` with tiny hyperparameters\n    "
    model_type = config_class.model_type
    config_source_file = inspect.getsourcefile(config_class)
    modeling_name = config_source_file.split(os.path.sep)[-1].replace('configuration_', '').replace('.py', '')
    try:
        print('Importing', model_type_to_module_name(model_type))
        module_name = model_type_to_module_name(model_type)
        if not modeling_name.startswith(module_name):
            raise ValueError(f"{modeling_name} doesn't start with {module_name}!")
        test_file = os.path.join('tests', 'models', module_name, f'test_modeling_{modeling_name}.py')
        models_to_model_testers = get_model_to_tester_mapping(test_file)
        model_tester_class = None
        tester_classes = []
        if model_class is not None:
            tester_classes = get_tester_classes_for_model(test_file, model_class)
        else:
            for _tester_classes in models_to_model_testers.values():
                tester_classes.extend(_tester_classes)
        if len(tester_classes) > 0:
            model_tester_class = sorted(tester_classes, key=lambda x: (len(x.__name__), x.__name__))[0]
    except ModuleNotFoundError:
        error = f'Tiny config not created for {model_type} - cannot find the testing module from the model name.'
        raise ValueError(error)
    if model_tester_class is None:
        error = f'Tiny config not created for {model_type} - no model tester is found in the testing module.'
        raise ValueError(error)
    model_tester = model_tester_class(parent=None, **model_tester_kwargs)
    if hasattr(model_tester, 'get_pipeline_config'):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, 'prepare_config_and_inputs'):
        return model_tester.prepare_config_and_inputs()[0]
    elif hasattr(model_tester, 'get_config'):
        return model_tester.get_config()
    else:
        error = f'Tiny config not created for {model_type} - the model tester {model_tester_class.__name__} lacks necessary method to create config.'
        raise ValueError(error)

def convert_tokenizer(tokenizer_fast: PreTrainedTokenizerFast):
    if False:
        print('Hello World!')
    new_tokenizer = tokenizer_fast.train_new_from_iterator(data['training_ds']['text'], TARGET_VOCAB_SIZE, show_progress=False)
    if not isinstance(new_tokenizer, LayoutLMv3TokenizerFast):
        new_tokenizer(data['testing_ds']['text'])
    return new_tokenizer

def convert_feature_extractor(feature_extractor, tiny_config):
    if False:
        i = 10
        return i + 15
    to_convert = False
    kwargs = {}
    if hasattr(tiny_config, 'image_size'):
        kwargs['size'] = tiny_config.image_size
        kwargs['crop_size'] = tiny_config.image_size
        to_convert = True
    elif hasattr(tiny_config, 'vision_config') and tiny_config.vision_config is not None and hasattr(tiny_config.vision_config, 'image_size'):
        kwargs['size'] = tiny_config.vision_config.image_size
        kwargs['crop_size'] = tiny_config.vision_config.image_size
        to_convert = True
    if hasattr(tiny_config, 'input_feat_per_channel'):
        kwargs['feature_size'] = tiny_config.input_feat_per_channel
        kwargs['num_mel_bins'] = tiny_config.input_feat_per_channel
        to_convert = True
    if to_convert:
        feature_extractor = feature_extractor.__class__(**kwargs)
    return feature_extractor

def convert_processors(processors, tiny_config, output_folder, result):
    if False:
        while True:
            i = 10
    'Change a processor to work with smaller inputs.\n\n    For tokenizers, we try to reduce their vocabulary size.\n\n    For feature extractor, we use smaller image size or change\n    other attributes using the values from `tiny_config`. See `convert_feature_extractor`.\n\n    This method should not fail: we catch the errors and put them in `result["warnings"]` with descriptive messages.\n    '

    def _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=False):
        if False:
            return 10
        'Set tokenizer(s) to `None` if the fast/slow tokenizers have different values for `vocab_size` or `length`.\n\n        If `keep_fast_tokenizer=True`, the fast tokenizer will be kept.\n        '
        if fast_tokenizer is not None and slow_tokenizer is not None:
            if fast_tokenizer.vocab_size != slow_tokenizer.vocab_size:
                warning_messagae = f'The fast/slow tokenizers ({fast_tokenizer.__class__.__name__}/{slow_tokenizer.__class__.__name__}) have different vocabulary size: fast_tokenizer.vocab_size = {fast_tokenizer.vocab_size} and slow_tokenizer.vocab_size = {slow_tokenizer.vocab_size}.'
                result['warnings'].append(warning_messagae)
                if not keep_fast_tokenizer:
                    fast_tokenizer = None
                slow_tokenizer = None
        if fast_tokenizer is not None and slow_tokenizer is not None:
            if len(fast_tokenizer) != len(slow_tokenizer):
                warning_messagae = f'The fast/slow tokenizers () have different length: len(fast_tokenizer) = {len(fast_tokenizer)} and len(slow_tokenizer) = {len(slow_tokenizer)}.'
                result['warnings'].append(warning_messagae)
                if not keep_fast_tokenizer:
                    fast_tokenizer = None
                slow_tokenizer = None
        return (fast_tokenizer, slow_tokenizer)
    tokenizers = []
    feature_extractors = []
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in tokenizers}:
                tokenizers.append(processor)
        elif isinstance(processor, BaseImageProcessor):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                feature_extractors.append(processor)
        elif isinstance(processor, FeatureExtractionMixin):
            if processor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                feature_extractors.append(processor)
        elif isinstance(processor, ProcessorMixin):
            if hasattr(processor, 'tokenizer'):
                if processor.tokenizer.__class__.__name__ not in {x.__class__.__name__ for x in tokenizers}:
                    tokenizers.append(processor.tokenizer)
            if hasattr(processor, 'image_processor'):
                if processor.image_processor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                    feature_extractors.append(processor.image_processor)
            elif hasattr(processor, 'feature_extractor'):
                if processor.feature_extractor.__class__.__name__ not in {x.__class__.__name__ for x in feature_extractors}:
                    feature_extractors.append(processor.feature_extractor)
    num_types = len({x.__class__.__name__ for x in feature_extractors})
    if num_types >= 2:
        raise ValueError(f'`feature_extractors` should contain at most 1 type, but it contains {num_types} types!')
    num_types = len({x.__class__.__name__.replace('Fast', '') for x in tokenizers})
    if num_types >= 2:
        raise ValueError(f'`tokenizers` should contain at most 1 tokenizer type, but it contains {num_types} types!')
    fast_tokenizer = None
    slow_tokenizer = None
    for tokenizer in tokenizers:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            fast_tokenizer = tokenizer
        else:
            slow_tokenizer = tokenizer
    (fast_tokenizer, slow_tokenizer) = _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=True)
    (original_fast_tokenizer, original_slow_tokenizer) = (fast_tokenizer, slow_tokenizer)
    if fast_tokenizer:
        try:
            if fast_tokenizer.vocab_size > TARGET_VOCAB_SIZE:
                fast_tokenizer = convert_tokenizer(fast_tokenizer)
        except Exception:
            result['warnings'].append((f'Failed to convert the fast tokenizer for {fast_tokenizer.__class__.__name__}.', traceback.format_exc()))
    if fast_tokenizer:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                fast_tokenizer.save_pretrained(tmpdir)
                try:
                    slow_tokenizer = AutoTokenizer.from_pretrained(tmpdir, use_fast=False)
                except Exception:
                    result['warnings'].append((f'Failed to load the slow tokenizer saved from {fast_tokenizer.__class__.__name__}.', traceback.format_exc()))
                    slow_tokenizer = None
        except Exception:
            result['warnings'].append((f'Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}.', traceback.format_exc()))
            fast_tokenizer = None
    (fast_tokenizer, slow_tokenizer) = _sanity_check(fast_tokenizer, slow_tokenizer, keep_fast_tokenizer=False)
    if original_fast_tokenizer is not None and fast_tokenizer is None or (original_slow_tokenizer is not None and slow_tokenizer is None):
        warning_messagae = 'There are some issues when converting the fast/slow tokenizers. The original tokenizers from the Hub  will be used instead.'
        result['warnings'].append(warning_messagae)
        fast_tokenizer = original_fast_tokenizer
        slow_tokenizer = original_slow_tokenizer
    if fast_tokenizer:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                fast_tokenizer.save_pretrained(tmpdir)
            except Exception:
                result['warnings'].append((f'Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}.', traceback.format_exc()))
                fast_tokenizer = None
    if slow_tokenizer:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                slow_tokenizer.save_pretrained(tmpdir)
            except Exception:
                result['warnings'].append((f'Failed to save the slow tokenizer for {slow_tokenizer.__class__.__name__}.', traceback.format_exc()))
                slow_tokenizer = None
    try:
        feature_extractors = [convert_feature_extractor(p, tiny_config) for p in feature_extractors]
    except Exception:
        result['warnings'].append(('Failed to convert feature extractors.', traceback.format_exc()))
        feature_extractors = []
    if hasattr(tiny_config, 'max_position_embeddings') and tiny_config.max_position_embeddings > 0:
        if fast_tokenizer is not None:
            if fast_tokenizer.__class__.__name__ in ['RobertaTokenizerFast', 'XLMRobertaTokenizerFast', 'LongformerTokenizerFast', 'MPNetTokenizerFast']:
                fast_tokenizer.model_max_length = tiny_config.max_position_embeddings - 2
            else:
                fast_tokenizer.model_max_length = tiny_config.max_position_embeddings
        if slow_tokenizer is not None:
            if slow_tokenizer.__class__.__name__ in ['RobertaTokenizer', 'XLMRobertaTokenizer', 'LongformerTokenizer', 'MPNetTokenizer']:
                slow_tokenizer.model_max_length = tiny_config.max_position_embeddings - 2
            else:
                slow_tokenizer.model_max_length = tiny_config.max_position_embeddings
    processors = [fast_tokenizer, slow_tokenizer] + feature_extractors
    processors = [p for p in processors if p is not None]
    for p in processors:
        p.save_pretrained(output_folder)
    return processors

def get_checkpoint_dir(output_dir, model_arch):
    if False:
        print('Hello World!')
    'Get framework-agnostic architecture name. Used to save all PT/TF/Flax models into the same directory.'
    arch_name = model_arch.__name__
    if arch_name.startswith('TF'):
        arch_name = arch_name[2:]
    elif arch_name.startswith('Flax'):
        arch_name = arch_name[4:]
    return os.path.join(output_dir, arch_name)

def build_model(model_arch, tiny_config, output_dir):
    if False:
        print('Hello World!')
    'Create and save a model for `model_arch`.\n\n    Also copy the set of processors to each model (under the same model type) output folder.\n    '
    checkpoint_dir = get_checkpoint_dir(output_dir, model_arch)
    processor_output_dir = os.path.join(output_dir, 'processors')
    if os.path.isdir(processor_output_dir):
        shutil.copytree(processor_output_dir, checkpoint_dir, dirs_exist_ok=True)
    tiny_config = copy.deepcopy(tiny_config)
    if any((model_arch.__name__.endswith(x) for x in ['ForCausalLM', 'LMHeadModel'])):
        tiny_config.is_encoder_decoder = False
        tiny_config.is_decoder = True
    model = model_arch(config=tiny_config)
    model.save_pretrained(checkpoint_dir)
    model.from_pretrained(checkpoint_dir)
    return model

def fill_result_with_error(result, error, trace, models_to_create):
    if False:
        while True:
            i = 10
    "Fill `result` with errors for all target model arch if we can't build processor"
    error = (error, trace)
    result['error'] = error
    for framework in FRAMEWORKS:
        if framework in models_to_create:
            result[framework] = {}
            for model_arch in models_to_create[framework]:
                result[framework][model_arch.__name__] = {'model': None, 'checkpoint': None, 'error': error}
    result['processor'] = {p.__class__.__name__: p.__class__.__name__ for p in result['processor'].values()}

def upload_model(model_dir, organization, token):
    if False:
        while True:
            i = 10
    'Upload the tiny models'
    arch_name = model_dir.split(os.path.sep)[-1]
    repo_name = f'tiny-random-{arch_name}'
    repo_id = f'{organization}/{repo_name}'
    repo_exist = False
    error = None
    try:
        create_repo(repo_id=repo_id, exist_ok=False, repo_type='model', token=token)
    except Exception as e:
        error = e
        if 'You already created' in str(e):
            error = None
            logger.warning('Remote repository exists and will be cloned.')
            repo_exist = True
            try:
                create_repo(repo_id=repo_id, exist_ok=True, repo_type='model', token=token)
            except Exception as e:
                error = e
    if error is not None:
        raise error
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(local_dir=tmpdir, clone_from=repo_id, token=token)
        repo.git_pull()
        shutil.copytree(model_dir, tmpdir, dirs_exist_ok=True)
        if repo_exist:
            hub_pr_url = upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type='model', commit_message=f'Update tiny models for {arch_name}', commit_description=f'Upload tiny models for {arch_name}', create_pr=True, token=token)
            logger.warning(f'PR open in {hub_pr_url}.')
        else:
            repo.git_add(auto_lfs_track=True)
            repo.git_commit(f'Upload tiny models for {arch_name}')
            repo.git_push(blocking=True)
            logger.warning(f'Tiny models {arch_name} pushed to {repo_id}.')

def build_composite_models(config_class, output_dir):
    if False:
        print('Hello World!')
    import tempfile
    from transformers import BertConfig, BertLMHeadModel, BertModel, BertTokenizer, BertTokenizerFast, EncoderDecoderModel, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast, SpeechEncoderDecoderModel, TFEncoderDecoderModel, TFVisionEncoderDecoderModel, TFVisionTextDualEncoderModel, VisionEncoderDecoderModel, VisionTextDualEncoderModel, ViTConfig, ViTFeatureExtractor, ViTModel, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2Processor
    result = {'error': None, 'warnings': []}
    if config_class.model_type == 'encoder-decoder':
        encoder_config_class = BertConfig
        decoder_config_class = BertConfig
        encoder_processor = (BertTokenizerFast, BertTokenizer)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = BertModel
        decoder_class = BertLMHeadModel
        model_class = EncoderDecoderModel
        tf_model_class = TFEncoderDecoderModel
    elif config_class.model_type == 'vision-encoder-decoder':
        encoder_config_class = ViTConfig
        decoder_config_class = GPT2Config
        encoder_processor = (ViTFeatureExtractor,)
        decoder_processor = (GPT2TokenizerFast, GPT2Tokenizer)
        encoder_class = ViTModel
        decoder_class = GPT2LMHeadModel
        model_class = VisionEncoderDecoderModel
        tf_model_class = TFVisionEncoderDecoderModel
    elif config_class.model_type == 'speech-encoder-decoder':
        encoder_config_class = Wav2Vec2Config
        decoder_config_class = BertConfig
        encoder_processor = (Wav2Vec2Processor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = Wav2Vec2Model
        decoder_class = BertLMHeadModel
        model_class = SpeechEncoderDecoderModel
        tf_model_class = None
    elif config_class.model_type == 'vision-text-dual-encoder':
        encoder_config_class = ViTConfig
        decoder_config_class = BertConfig
        encoder_processor = (ViTFeatureExtractor,)
        decoder_processor = (BertTokenizerFast, BertTokenizer)
        encoder_class = ViTModel
        decoder_class = BertModel
        model_class = VisionTextDualEncoderModel
        tf_model_class = TFVisionTextDualEncoderModel
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            models_to_create = {'processor': encoder_processor, 'pytorch': (encoder_class,), 'tensorflow': []}
            encoder_output_dir = os.path.join(tmpdir, 'encoder')
            build(encoder_config_class, models_to_create, encoder_output_dir)
            models_to_create = {'processor': decoder_processor, 'pytorch': (decoder_class,), 'tensorflow': []}
            decoder_output_dir = os.path.join(tmpdir, 'decoder')
            build(decoder_config_class, models_to_create, decoder_output_dir)
            encoder_path = os.path.join(encoder_output_dir, encoder_class.__name__)
            decoder_path = os.path.join(decoder_output_dir, decoder_class.__name__)
            if config_class.model_type != 'vision-text-dual-encoder':
                decoder_config = decoder_config_class.from_pretrained(decoder_path)
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
                model = model_class.from_encoder_decoder_pretrained(encoder_path, decoder_path, decoder_config=decoder_config)
            elif config_class.model_type == 'vision-text-dual-encoder':
                model = model_class.from_vision_text_pretrained(encoder_path, decoder_path)
            model_path = os.path.join(output_dir, f'{model_class.__name__}-{encoder_config_class.model_type}-{decoder_config_class.model_type}')
            model.save_pretrained(model_path)
            if tf_model_class is not None:
                model = tf_model_class.from_pretrained(model_path)
                model.save_pretrained(model_path)
            encoder_processor_path = os.path.join(encoder_output_dir, 'processors')
            decoder_processor_path = os.path.join(decoder_output_dir, 'processors')
            if os.path.isdir(encoder_processor_path):
                shutil.copytree(encoder_processor_path, model_path, dirs_exist_ok=True)
            if os.path.isdir(decoder_processor_path):
                shutil.copytree(decoder_processor_path, model_path, dirs_exist_ok=True)
            result['processor'] = {x.__name__: x.__name__ for x in encoder_processor + decoder_processor}
            result['pytorch'] = {model_class.__name__: {'model': model_class.__name__, 'checkpoint': model_path}}
            result['tensorflow'] = {}
            if tf_model_class is not None:
                result['tensorflow'] = {tf_model_class.__name__: {'model': tf_model_class.__name__, 'checkpoint': model_path}}
        except Exception:
            result['error'] = (f'Failed to build models for {config_class.__name__}.', traceback.format_exc())
    if not result['error']:
        del result['error']
    if not result['warnings']:
        del result['warnings']
    return result

def get_token_id_from_tokenizer(token_id_name, tokenizer, original_token_id):
    if False:
        print('Hello World!')
    'Use `tokenizer` to get the values of `bos_token_id`, `eos_token_ids`, etc.\n\n    The argument `token_id_name` should be a string ending with `_token_id`, and `original_token_id` should be an\n    integer that will be return if `tokenizer` has no token corresponding to `token_id_name`.\n    '
    token_id = original_token_id
    if not token_id_name.endswith('_token_id'):
        raise ValueError(f"`token_id_name` is {token_id_name}, which doesn't end with `_token_id`!")
    token = getattr(tokenizer, token_id_name.replace('_token_id', '_token'), None)
    if token is not None:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            token_id = tokenizer._convert_token_to_id_with_added_voc(token)
        else:
            token_id = tokenizer._convert_token_to_id(token)
    return token_id

def get_config_overrides(config_class, processors):
    if False:
        i = 10
        return i + 15
    if config_class.__name__ == 'BarkConfig':
        return {}
    config_overrides = {}
    tokenizer = None
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerFast):
            tokenizer = processor
            break
        elif isinstance(processor, PreTrainedTokenizer):
            tokenizer = processor
    if tokenizer is None:
        return config_overrides
    vocab_size = len(tokenizer)
    if config_class.__name__ == 'GPTSanJapaneseConfig':
        vocab_size += 2
    config_overrides['vocab_size'] = vocab_size
    model_tester_kwargs = {'vocab_size': vocab_size}
    if config_class.__name__ in ['AlignConfig', 'AltCLIPConfig', 'ChineseCLIPConfig', 'CLIPSegConfig', 'ClapConfig', 'CLIPConfig', 'GroupViTConfig', 'OwlViTConfig', 'XCLIPConfig', 'FlavaConfig', 'BlipConfig', 'Blip2Config']:
        del model_tester_kwargs['vocab_size']
        model_tester_kwargs['text_kwargs'] = {'vocab_size': vocab_size}
    elif config_class.__name__ == 'FSMTConfig':
        del model_tester_kwargs['vocab_size']
        model_tester_kwargs['src_vocab_size'] = tokenizer.src_vocab_size
        model_tester_kwargs['tgt_vocab_size'] = tokenizer.tgt_vocab_size
    _tiny_config = get_tiny_config(config_class, **model_tester_kwargs)
    if hasattr(_tiny_config, 'text_config'):
        _tiny_config = _tiny_config.text_config
    for attr in dir(_tiny_config):
        if attr.endswith('_token_id'):
            token_id = getattr(_tiny_config, attr)
            if token_id is not None:
                token_id = get_token_id_from_tokenizer(attr, tokenizer, original_token_id=token_id)
                config_overrides[attr] = token_id
    if config_class.__name__ == 'FSMTConfig':
        config_overrides['src_vocab_size'] = tokenizer.src_vocab_size
        config_overrides['tgt_vocab_size'] = tokenizer.tgt_vocab_size
        config_overrides['decoder'] = configuration_fsmt.DecoderConfig(vocab_size=tokenizer.tgt_vocab_size, bos_token_id=config_overrides['eos_token_id'])
    return config_overrides

def build(config_class, models_to_create, output_dir):
    if False:
        i = 10
        return i + 15
    'Create all models for a certain model type.\n\n    Args:\n        config_class (`PretrainedConfig`):\n            A subclass of `PretrainedConfig` that is used to determine `models_to_create`.\n        models_to_create (`dict`):\n            A dictionary containing the processor/model classes that we want to create the instances. These models are\n            of the same model type which is associated to `config_class`.\n        output_dir (`str`):\n            The directory to save all the checkpoints. Each model architecture will be saved in a subdirectory under\n            it. Models in different frameworks with the same architecture will be saved in the same subdirectory.\n    '
    if data['training_ds'] is None or data['testing_ds'] is None:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
        data['training_ds'] = ds['train']
        data['testing_ds'] = ds['test']
    if config_class.model_type in ['encoder-decoder', 'vision-encoder-decoder', 'speech-encoder-decoder', 'vision-text-dual-encoder']:
        return build_composite_models(config_class, output_dir)
    result = {k: {} for k in models_to_create}
    result['error'] = None
    result['warnings'] = []
    processor_classes = models_to_create['processor']
    if len(processor_classes) == 0:
        error = f'No processor class could be found in {config_class.__name__}.'
        fill_result_with_error(result, error, None, models_to_create)
        logger.error(result['error'][0])
        return result
    for processor_class in processor_classes:
        try:
            processor = build_processor(config_class, processor_class, allow_no_checkpoint=True)
            if processor is not None:
                result['processor'][processor_class] = processor
        except Exception:
            error = f'Failed to build processor for {processor_class.__name__}.'
            trace = traceback.format_exc()
            fill_result_with_error(result, error, trace, models_to_create)
            logger.error(result['error'][0])
            return result
    if len(result['processor']) == 0:
        error = f'No processor could be built for {config_class.__name__}.'
        fill_result_with_error(result, error, None, models_to_create)
        logger.error(result['error'][0])
        return result
    try:
        tiny_config = get_tiny_config(config_class)
    except Exception as e:
        error = f'Failed to get tiny config for {config_class.__name__}: {e}'
        trace = traceback.format_exc()
        fill_result_with_error(result, error, trace, models_to_create)
        logger.error(result['error'][0])
        return result
    processors = list(result['processor'].values())
    processor_output_folder = os.path.join(output_dir, 'processors')
    try:
        processors = convert_processors(processors, tiny_config, processor_output_folder, result)
    except Exception:
        error = 'Failed to convert the processors.'
        trace = traceback.format_exc()
        result['warnings'].append((error, trace))
    if len(processors) == 0:
        error = f'No processor is returned by `convert_processors` for {config_class.__name__}.'
        fill_result_with_error(result, error, None, models_to_create)
        logger.error(result['error'][0])
        return result
    try:
        config_overrides = get_config_overrides(config_class, processors)
    except Exception as e:
        error = f'Failure occurs while calling `get_config_overrides`: {e}'
        trace = traceback.format_exc()
        fill_result_with_error(result, error, trace, models_to_create)
        logger.error(result['error'][0])
        return result
    if 'vocab_size' in config_overrides:
        result['vocab_size'] = config_overrides['vocab_size']
    for (k, v) in config_overrides.items():
        if hasattr(tiny_config, k):
            setattr(tiny_config, k, v)
        elif hasattr(tiny_config, 'text_config') and tiny_config.text_config is not None and hasattr(tiny_config.text_config, k):
            setattr(tiny_config.text_config, k, v)
            if hasattr(tiny_config, 'text_config_dict'):
                tiny_config.text_config_dict[k] = v
    if result['warnings']:
        logger.warning(result['warnings'][0][0])
    result['processor'] = {type(p).__name__: p.__class__.__name__ for p in processors}
    for pytorch_arch in models_to_create['pytorch']:
        result['pytorch'][pytorch_arch.__name__] = {}
        error = None
        try:
            model = build_model(pytorch_arch, tiny_config, output_dir=output_dir)
        except Exception as e:
            model = None
            error = f'Failed to create the pytorch model for {pytorch_arch}: {e}'
            trace = traceback.format_exc()
        result['pytorch'][pytorch_arch.__name__]['model'] = model.__class__.__name__ if model is not None else None
        result['pytorch'][pytorch_arch.__name__]['checkpoint'] = get_checkpoint_dir(output_dir, pytorch_arch) if model is not None else None
        if error is not None:
            result['pytorch'][pytorch_arch.__name__]['error'] = (error, trace)
            logger.error(f'{pytorch_arch.__name__}: {error}')
    for tensorflow_arch in models_to_create['tensorflow']:
        pt_arch_name = tensorflow_arch.__name__[2:]
        pt_arch = getattr(transformers_module, pt_arch_name)
        result['tensorflow'][tensorflow_arch.__name__] = {}
        error = None
        if pt_arch.__name__ in result['pytorch'] and result['pytorch'][pt_arch.__name__]['checkpoint'] is not None:
            ckpt = get_checkpoint_dir(output_dir, pt_arch)
            try:
                model = tensorflow_arch.from_pretrained(ckpt)
                model.save_pretrained(ckpt)
            except Exception as e:
                model = None
                error = f'Failed to convert the pytorch model to the tensorflow model for {pt_arch}: {e}'
                trace = traceback.format_exc()
        else:
            try:
                model = build_model(tensorflow_arch, tiny_config, output_dir=output_dir)
            except Exception as e:
                model = None
                error = f'Failed to create the tensorflow model for {tensorflow_arch}: {e}'
                trace = traceback.format_exc()
        result['tensorflow'][tensorflow_arch.__name__]['model'] = model.__class__.__name__ if model is not None else None
        result['tensorflow'][tensorflow_arch.__name__]['checkpoint'] = get_checkpoint_dir(output_dir, tensorflow_arch) if model is not None else None
        if error is not None:
            result['tensorflow'][tensorflow_arch.__name__]['error'] = (error, trace)
            logger.error(f'{tensorflow_arch.__name__}: {error}')
    if not result['error']:
        del result['error']
    if not result['warnings']:
        del result['warnings']
    return result

def build_tiny_model_summary(results, organization=None, token=None):
    if False:
        i = 10
        return i + 15
    'Build a summary: a dictionary of the form\n    {\n      model architecture name:\n        {\n          "tokenizer_classes": [...],\n          "processor_classes": [...],\n          "model_classes": [...],\n        }\n      ..\n    }\n    '
    tiny_model_summary = {}
    for config_name in results:
        processors = [key for (key, value) in results[config_name]['processor'].items()]
        tokenizer_classes = sorted([x for x in processors if x.endswith('TokenizerFast') or x.endswith('Tokenizer')])
        processor_classes = sorted([x for x in processors if x not in tokenizer_classes])
        for framework in FRAMEWORKS:
            if framework not in results[config_name]:
                continue
            for arch_name in results[config_name][framework]:
                model_classes = [arch_name]
                base_arch_name = arch_name[2:] if arch_name.startswith('TF') else arch_name
                if results[config_name][framework][arch_name]['model'] is None:
                    model_classes = []
                if base_arch_name not in tiny_model_summary:
                    tiny_model_summary[base_arch_name] = {}
                tiny_model_summary[base_arch_name].update({'tokenizer_classes': tokenizer_classes, 'processor_classes': processor_classes})
                tiny_model_summary[base_arch_name]['model_classes'] = sorted(tiny_model_summary[base_arch_name].get('model_classes', []) + model_classes)
                if organization is not None:
                    repo_name = f'tiny-random-{base_arch_name}'
                    if base_arch_name in COMPOSITE_MODELS:
                        repo_name = f'tiny-random-{COMPOSITE_MODELS[base_arch_name]}'
                    repo_id = f'{organization}/{repo_name}'
                    try:
                        commit_hash = hf_api.repo_info(repo_id, token=token).sha
                    except Exception:
                        logger.warning(f'Failed to get information for {repo_id}.\n{traceback.format_exc()}')
                        del tiny_model_summary[base_arch_name]
                        continue
                    tiny_model_summary[base_arch_name]['sha'] = commit_hash
    return tiny_model_summary

def build_failed_report(results, include_warning=True):
    if False:
        i = 10
        return i + 15
    failed_results = {}
    for config_name in results:
        if 'error' in results[config_name]:
            if config_name not in failed_results:
                failed_results[config_name] = {}
            failed_results[config_name] = {'error': results[config_name]['error']}
        if include_warning and 'warnings' in results[config_name]:
            if config_name not in failed_results:
                failed_results[config_name] = {}
            failed_results[config_name]['warnings'] = results[config_name]['warnings']
        for framework in FRAMEWORKS:
            if framework not in results[config_name]:
                continue
            for arch_name in results[config_name][framework]:
                if 'error' in results[config_name][framework][arch_name]:
                    if config_name not in failed_results:
                        failed_results[config_name] = {}
                    if framework not in failed_results[config_name]:
                        failed_results[config_name][framework] = {}
                    if arch_name not in failed_results[config_name][framework]:
                        failed_results[config_name][framework][arch_name] = {}
                    error = results[config_name][framework][arch_name]['error']
                    failed_results[config_name][framework][arch_name]['error'] = error
    return failed_results

def build_simple_report(results):
    if False:
        return 10
    text = ''
    failed_text = ''
    for config_name in results:
        for framework in FRAMEWORKS:
            if framework not in results[config_name]:
                continue
            for arch_name in results[config_name][framework]:
                if 'error' in results[config_name][framework][arch_name]:
                    result = results[config_name][framework][arch_name]['error']
                    failed_text += f'{arch_name}: {result[0]}\n'
                else:
                    result = ('OK',)
                text += f'{arch_name}: {result[0]}\n'
    return (text, failed_text)

def update_tiny_model_summary_file(report_path):
    if False:
        i = 10
        return i + 15
    with open(os.path.join(report_path, 'tiny_model_summary.json')) as fp:
        new_data = json.load(fp)
    with open('tests/utils/tiny_model_summary.json') as fp:
        data = json.load(fp)
    for (key, value) in new_data.items():
        if key not in data:
            data[key] = value
        else:
            for attr in ['tokenizer_classes', 'processor_classes', 'model_classes']:
                data[key][attr].extend(value[attr])
            new_sha = value.get('sha', None)
            if new_sha is not None:
                data[key]['sha'] = new_sha
    updated_data = {}
    for key in sorted(data.keys()):
        updated_data[key] = {}
        for (attr, value) in data[key].items():
            updated_data[key][attr] = sorted(set(value)) if attr != 'sha' else value
    with open(os.path.join(report_path, 'updated_tiny_model_summary.json'), 'w') as fp:
        json.dump(updated_data, fp, indent=4, ensure_ascii=False)

def create_tiny_models(output_path, all, model_types, models_to_skip, no_check, upload, organization, token, num_workers=1):
    if False:
        i = 10
        return i + 15
    clone_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if os.getcwd() != clone_path:
        raise ValueError(f'This script should be run from the root of the clone of `transformers` {clone_path}')
    report_path = os.path.join(output_path, 'reports')
    os.makedirs(report_path)
    _pytorch_arch_mappings = [x for x in dir(transformers_module) if x.startswith('MODEL_') and x.endswith('_MAPPING') and (x != 'MODEL_NAMES_MAPPING')]
    _tensorflow_arch_mappings = [x for x in dir(transformers_module) if x.startswith('TF_MODEL_') and x.endswith('_MAPPING')]
    pytorch_arch_mappings = [getattr(transformers_module, x) for x in _pytorch_arch_mappings]
    tensorflow_arch_mappings = [getattr(transformers_module, x) for x in _tensorflow_arch_mappings]
    config_classes = CONFIG_MAPPING.values()
    if not all:
        config_classes = [CONFIG_MAPPING[model_type] for model_type in model_types]
    processor_type_map = {c: get_processor_types_from_config_class(c) for c in config_classes}
    to_create = {}
    for c in config_classes:
        processors = processor_type_map[c]
        models = get_architectures_from_config_class(c, pytorch_arch_mappings, models_to_skip)
        tf_models = get_architectures_from_config_class(c, tensorflow_arch_mappings, models_to_skip)
        if len(models) + len(tf_models) > 0:
            to_create[c] = {'processor': processors, 'pytorch': models, 'tensorflow': tf_models}
    results = {}
    if num_workers <= 1:
        for (c, models_to_create) in list(to_create.items()):
            print(f'Create models for {c.__name__} ...')
            result = build(c, models_to_create, output_dir=os.path.join(output_path, c.model_type))
            results[c.__name__] = result
            print('=' * 40)
    else:
        all_build_args = []
        for (c, models_to_create) in list(to_create.items()):
            all_build_args.append((c, models_to_create, os.path.join(output_path, c.model_type)))
        with multiprocessing.Pool() as pool:
            results = pool.starmap(build, all_build_args)
            results = {buid_args[0].__name__: result for (buid_args, result) in zip(all_build_args, results)}
    if upload:
        if organization is None:
            raise ValueError('The argument `organization` could not be `None`. No model is uploaded')
        to_upload = []
        for model_type in os.listdir(output_path):
            if model_type == 'reports':
                continue
            for arch in os.listdir(os.path.join(output_path, model_type)):
                if arch == 'processors':
                    continue
                to_upload.append(os.path.join(output_path, model_type, arch))
        to_upload = sorted(to_upload)
        upload_results = {}
        if len(to_upload) > 0:
            for model_dir in to_upload:
                try:
                    upload_model(model_dir, organization, token)
                except Exception as e:
                    error = f'Failed to upload {model_dir}. {e.__class__.__name__}: {e}'
                    logger.error(error)
                    upload_results[model_dir] = error
        with open(os.path.join(report_path, 'failed_uploads.json'), 'w') as fp:
            json.dump(upload_results, fp, indent=4)
    tiny_model_summary = build_tiny_model_summary(results, organization=organization, token=token)
    with open(os.path.join(report_path, 'tiny_model_summary.json'), 'w') as fp:
        json.dump(tiny_model_summary, fp, indent=4)
    with open(os.path.join(report_path, 'tiny_model_creation_report.json'), 'w') as fp:
        json.dump(results, fp, indent=4)
    failed_results = build_failed_report(results)
    with open(os.path.join(report_path, 'failed_report.json'), 'w') as fp:
        json.dump(failed_results, fp, indent=4)
    (simple_report, failed_report) = build_simple_report(results)
    with open(os.path.join(report_path, 'simple_report.txt'), 'w') as fp:
        fp.write(simple_report)
    with open(os.path.join(report_path, 'simple_failed_report.txt'), 'w') as fp:
        fp.write(failed_report)
    update_tiny_model_summary_file(report_path=os.path.join(output_path, 'reports'))
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    def list_str(values):
        if False:
            return 10
        return values.split(',')
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Will create all tiny models.')
    parser.add_argument('--no_check', action='store_true', help='If set, will not check the validity of architectures. Use with caution.')
    parser.add_argument('-m', '--model_types', type=list_str, help='Comma-separated list of model type(s) from which the tiny models will be created.')
    parser.add_argument('--models_to_skip', type=list_str, help="Comma-separated list of model class names(s) from which the tiny models won't be created.\nThis is usually the list of model classes that have their tiny versions already uploaded to the Hub.")
    parser.add_argument('--upload', action='store_true', help='If to upload the created tiny models to the Hub.')
    parser.add_argument('--organization', default=None, type=str, help='The organization on the Hub to which the tiny models will be uploaded.')
    parser.add_argument('--token', default=None, type=str, help='A valid authentication token for HuggingFace Hub with write access.')
    parser.add_argument('output_path', type=Path, help='Path indicating where to store generated model.')
    parser.add_argument('--num_workers', default=1, type=int, help='The number of workers to run.')
    args = parser.parse_args()
    if not args.all and (not args.model_types):
        raise ValueError('Please provide at least one model type or pass `--all` to export all architectures.')
    create_tiny_models(args.output_path, args.all, args.model_types, args.models_to_skip, args.no_check, args.upload, args.organization, args.token, args.num_workers)