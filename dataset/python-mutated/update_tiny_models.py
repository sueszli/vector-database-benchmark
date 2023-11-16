"""A script running `create_dummy_models.py` with a pre-defined set of arguments.

This file is intended to be used in a CI workflow file without the need of specifying arguments. It creates and uploads
tiny models for all model classes (if their tiny versions are not on the Hub yet), as well as produces an updated
version of `tests/utils/tiny_model_summary.json`. That updated file should be merged into the `main` branch of
`transformers` so the pipeline testing will use the latest created/updated tiny models.
"""
import argparse
import copy
import json
import multiprocessing
import os
import time
from create_dummy_models import COMPOSITE_MODELS, create_tiny_models
from huggingface_hub import ModelFilter, hf_api
import transformers
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor

def get_all_model_names():
    if False:
        i = 10
        return i + 15
    model_names = set()
    for module_name in ['modeling_auto', 'modeling_tf_auto', 'modeling_flax_auto']:
        module = getattr(transformers.models.auto, module_name, None)
        if module is None:
            continue
        mapping_names = [x for x in dir(module) if x.endswith('_MAPPING_NAMES') and (x.startswith('MODEL_') or x.startswith('TF_MODEL_') or x.startswith('FLAX_MODEL_'))]
        for name in mapping_names:
            mapping = getattr(module, name)
            if mapping is not None:
                for v in mapping.values():
                    if isinstance(v, (list, tuple)):
                        model_names.update(v)
                    elif isinstance(v, str):
                        model_names.add(v)
    return sorted(model_names)

def get_tiny_model_names_from_repo():
    if False:
        for i in range(10):
            print('nop')
    model_names = set(get_all_model_names())
    with open('tests/utils/tiny_model_summary.json') as fp:
        tiny_model_info = json.load(fp)
    tiny_models_names = set()
    for model_base_name in tiny_model_info:
        tiny_models_names.update(tiny_model_info[model_base_name]['model_classes'])
    not_on_hub = model_names.difference(tiny_models_names)
    for model_name in copy.copy(tiny_models_names):
        if not model_name.startswith('TF') and f'TF{model_name}' in not_on_hub:
            tiny_models_names.remove(model_name)
        elif model_name.startswith('TF') and model_name[2:] in not_on_hub:
            tiny_models_names.remove(model_name)
    return sorted(tiny_models_names)

def get_tiny_model_summary_from_hub(output_path):
    if False:
        print('Hello World!')
    special_models = COMPOSITE_MODELS.values()
    model_names = get_all_model_names()
    models = hf_api.list_models(filter=ModelFilter(author='hf-internal-testing'))
    _models = set()
    for x in models:
        model = x.modelId
        (org, model) = model.split('/')
        if not model.startswith('tiny-random-'):
            continue
        model = model.replace('tiny-random-', '')
        if not model[0].isupper():
            continue
        if model not in model_names and model not in special_models:
            continue
        _models.add(model)
    models = sorted(_models)
    summary = {}
    for model in models:
        repo_id = f'hf-internal-testing/tiny-random-{model}'
        model = model.split('-')[0]
        try:
            repo_info = hf_api.repo_info(repo_id)
            content = {'tokenizer_classes': set(), 'processor_classes': set(), 'model_classes': set(), 'sha': repo_info.sha}
        except Exception:
            continue
        try:
            time.sleep(1)
            tokenizer_fast = AutoTokenizer.from_pretrained(repo_id)
            content['tokenizer_classes'].add(tokenizer_fast.__class__.__name__)
        except Exception:
            pass
        try:
            time.sleep(1)
            tokenizer_slow = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
            content['tokenizer_classes'].add(tokenizer_slow.__class__.__name__)
        except Exception:
            pass
        try:
            time.sleep(1)
            img_p = AutoImageProcessor.from_pretrained(repo_id)
            content['processor_classes'].add(img_p.__class__.__name__)
        except Exception:
            pass
        try:
            time.sleep(1)
            feat_p = AutoFeatureExtractor.from_pretrained(repo_id)
            if not isinstance(feat_p, BaseImageProcessor):
                content['processor_classes'].add(feat_p.__class__.__name__)
        except Exception:
            pass
        try:
            time.sleep(1)
            model_class = getattr(transformers, model)
            m = model_class.from_pretrained(repo_id)
            content['model_classes'].add(m.__class__.__name__)
        except Exception:
            pass
        try:
            time.sleep(1)
            model_class = getattr(transformers, f'TF{model}')
            m = model_class.from_pretrained(repo_id)
            content['model_classes'].add(m.__class__.__name__)
        except Exception:
            pass
        content['tokenizer_classes'] = sorted(content['tokenizer_classes'])
        content['processor_classes'] = sorted(content['processor_classes'])
        content['model_classes'] = sorted(content['model_classes'])
        summary[model] = content
        with open(os.path.join(output_path, 'hub_tiny_model_summary.json'), 'w') as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=1, type=int, help='The number of workers to run.')
    args = parser.parse_args()
    multiprocessing.set_start_method('spawn')
    output_path = 'tiny_models'
    all = True
    model_types = None
    models_to_skip = get_tiny_model_names_from_repo()
    no_check = True
    upload = True
    organization = 'hf-internal-testing'
    create_tiny_models(output_path, all, model_types, models_to_skip, no_check, upload, organization, token=os.environ.get('TOKEN', None), num_workers=args.num_workers)