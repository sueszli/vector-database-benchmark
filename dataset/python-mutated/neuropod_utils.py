import importlib.util
import logging
import os
import tempfile
from typing import Any, Dict, List
import torch
from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import NAME
from ludwig.types import ModelConfigDict
from ludwig.utils.fs_utils import open_file
logger = logging.getLogger(__name__)
INFERENCE_MODULE_TEMPLATE = '\nfrom typing import Any, Dict, List, Tuple, Union\nimport torch\nfrom ludwig.utils.types import TorchscriptPreprocessingInput\n\nclass GeneratedInferenceModule(torch.nn.Module):\n    def __init__(self, inference_module):\n        super().__init__()\n        self.inference_module = inference_module\n\n    def forward(self, {input_signature}):\n        inputs: Dict[str, TorchscriptPreprocessingInput] = {input_dict}\n        results = self.inference_module(inputs)\n        return {output_dicts}\n'

def _get_input_signature(config: ModelConfigDict) -> str:
    if False:
        print('Hello World!')
    args = []
    for feature in config['input_features']:
        name = feature[NAME]
        args.append(f'{name}: TorchscriptPreprocessingInput')
    return ', '.join(args)

def _get_input_dict(config: ModelConfigDict) -> str:
    if False:
        return 10
    elems = []
    for feature in config['input_features']:
        name = feature[NAME]
        elems.append(f'"{name}": {name}')
    return '{' + ', '.join(elems) + '}'

def _get_output_dicts(config: ModelConfigDict) -> str:
    if False:
        return 10
    results = []
    for feature in config['output_features']:
        name = feature[NAME]
        results.append(f'"{name}": results["{name}"]["predictions"]')
    return '{' + ', '.join(results) + '}'

@DeveloperAPI
def generate_neuropod_torchscript(model: LudwigModel):
    if False:
        while True:
            i = 10
    config = model.config
    inference_module = model.to_torchscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, 'generated.py')
        with open_file(ts_path, 'w') as f:
            f.write(INFERENCE_MODULE_TEMPLATE.format(input_signature=_get_input_signature(config), input_dict=_get_input_dict(config), output_dicts=_get_output_dicts(config)))
        spec = importlib.util.spec_from_file_location('generated.ts', ts_path)
        gen_ts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_ts)
        gen_module = gen_ts.GeneratedInferenceModule(inference_module)
        scripted_module = torch.jit.script(gen_module)
    return scripted_module

def _get_input_spec(model: LudwigModel) -> List[Dict[str, Any]]:
    if False:
        return 10
    spec = []
    for (feature_name, feature) in model.model.input_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append({'name': feature.feature_name, 'dtype': feature.get_preproc_input_dtype(metadata), 'shape': ('batch_size',)})
    return spec

def _get_output_spec(model: LudwigModel) -> List[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    spec = []
    for (feature_name, feature) in model.model.output_features.items():
        metadata = model.training_set_metadata[feature_name]
        spec.append({'name': feature.feature_name, 'dtype': feature.get_postproc_output_dtype(metadata), 'shape': ('batch_size',)})
    return spec

@DeveloperAPI
def export_neuropod(model: LudwigModel, neuropod_path: str, neuropod_model_name='ludwig_model'):
    if False:
        i = 10
        return i + 15
    try:
        from neuropod.backends.torchscript.packager import create_torchscript_neuropod
    except ImportError:
        raise RuntimeError('The "neuropod" package is not installed in your environment.')
    model_ts = generate_neuropod_torchscript(model)
    create_torchscript_neuropod(neuropod_path=neuropod_path, model_name=neuropod_model_name, module=model_ts, input_spec=_get_input_spec(model), output_spec=_get_output_spec(model))