import importlib
import logging
import os
import re
import subprocess
import sys
import warnings
import torch
from common import BenchmarkRunner, download_retry_decorator, main
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

def pip_install(package):
    if False:
        i = 10
        return i + 15
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
try:
    importlib.import_module('timm')
except ModuleNotFoundError:
    print('Installing Pytorch Image Models...')
    pip_install('git+https://github.com/rwightman/pytorch-image-models')
finally:
    from timm import __version__ as timmversion
    from timm.data import resolve_data_config
    from timm.models import create_model
TIMM_MODELS = dict()
filename = os.path.join(os.path.dirname(__file__), 'timm_models_list.txt')
with open(filename) as fh:
    lines = fh.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        (model_name, batch_size) = line.split(' ')
        TIMM_MODELS[model_name] = int(batch_size)
BATCH_SIZE_DIVISORS = {'beit_base_patch16_224': 2, 'cait_m36_384': 8, 'convit_base': 2, 'convmixer_768_32': 2, 'convnext_base': 2, 'cspdarknet53': 2, 'deit_base_distilled_patch16_224': 2, 'dpn107': 2, 'gluon_xception65': 2, 'mobilevit_s': 2, 'pit_b_224': 2, 'pnasnet5large': 2, 'poolformer_m36': 2, 'res2net101_26w_4s': 2, 'resnest101e': 2, 'sebotnet33ts_256': 2, 'swin_base_patch4_window7_224': 2, 'swsl_resnext101_32x16d': 2, 'twins_pcpvt_base': 2, 'vit_base_patch16_224': 2, 'volo_d1_224': 2, 'jx_nest_base': 4, 'xcit_large_24_p8_224': 4}
REQUIRE_HIGHER_TOLERANCE = set('sebotnet33ts_256')
SCALED_COMPUTE_LOSS = {'ese_vovnet19b_dw', 'fbnetc_100', 'mnasnet_100', 'mobilevit_s', 'sebotnet33ts_256'}
FORCE_AMP_FOR_FP16_BF16_MODELS = {'convit_base', 'xcit_large_24_p8_224'}

def refresh_model_names():
    if False:
        return 10
    import glob
    from timm.models import list_models

    def read_models_from_docs():
        if False:
            return 10
        models = set()
        for fn in glob.glob('../pytorch-image-models/docs/models/*.md'):
            with open(fn) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.startswith('model = timm.create_model('):
                        continue
                    model = line.split("'")[1]
                    models.add(model)
        return models

    def get_family_name(name):
        if False:
            i = 10
            return i + 15
        known_families = ['darknet', 'densenet', 'dla', 'dpn', 'ecaresnet', 'halo', 'regnet', 'efficientnet', 'deit', 'mobilevit', 'mnasnet', 'convnext', 'resnet', 'resnest', 'resnext', 'selecsls', 'vgg', 'xception']
        for known_family in known_families:
            if known_family in name:
                return known_family
        if name.startswith('gluon_'):
            return 'gluon_' + name.split('_')[1]
        return name.split('_')[0]

    def populate_family(models):
        if False:
            while True:
                i = 10
        family = dict()
        for model_name in models:
            family_name = get_family_name(model_name)
            if family_name not in family:
                family[family_name] = []
            family[family_name].append(model_name)
        return family
    docs_models = read_models_from_docs()
    all_models = list_models(pretrained=True, exclude_filters=['*in21k'])
    all_models_family = populate_family(all_models)
    docs_models_family = populate_family(docs_models)
    for key in docs_models_family:
        del all_models_family[key]
    chosen_models = set()
    for value in docs_models_family.values():
        chosen_models.add(value[0])
    for (key, value) in all_models_family.items():
        chosen_models.add(value[0])
    filename = 'timm_models_list.txt'
    if os.path.exists('benchmarks'):
        filename = 'benchmarks/' + filename
    with open(filename, 'w') as fw:
        for model_name in sorted(chosen_models):
            fw.write(model_name + '\n')

class TimmRunner(BenchmarkRunner):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.suite_name = 'timm_models'

    @property
    def force_amp_for_fp16_bf16_models(self):
        if False:
            print('Hello World!')
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @property
    def force_fp16_for_bf16_models(self):
        if False:
            i = 10
            return i + 15
        return set()

    @download_retry_decorator
    def _download_model(self, model_name):
        if False:
            while True:
                i = 10
        model = create_model(model_name, in_chans=3, scriptable=False, num_classes=None, drop_rate=0.0, drop_path_rate=None, drop_block_rate=None, pretrained=True)
        return model

    def load_model(self, device, model_name, batch_size=None, extra_args=None):
        if False:
            print('Hello World!')
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError('Activation checkpointing not implemented for Timm models')
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        channels_last = self._args.channels_last
        model = self._download_model(model_name)
        if model is None:
            raise RuntimeError(f"Failed to load model '{model_name}'")
        model.to(device=device, memory_format=torch.channels_last if channels_last else None)
        self.num_classes = model.num_classes
        data_config = resolve_data_config(vars(self._args) if timmversion >= '0.8.0' else self._args, model=model, use_test_size=not is_training)
        input_size = data_config['input_size']
        recorded_batch_size = TIMM_MODELS[model_name]
        if model_name in BATCH_SIZE_DIVISORS:
            recorded_batch_size = max(int(recorded_batch_size / BATCH_SIZE_DIVISORS[model_name]), 1)
        batch_size = batch_size or recorded_batch_size
        torch.manual_seed(1337)
        input_tensor = torch.randint(256, size=(batch_size,) + input_size, device=device).to(dtype=torch.float32)
        mean = torch.mean(input_tensor)
        std_dev = torch.std(input_tensor)
        example_inputs = (input_tensor - mean) / std_dev
        if channels_last:
            example_inputs = example_inputs.contiguous(memory_format=torch.channels_last)
        example_inputs = [example_inputs]
        self.target = self._gen_target(batch_size, device)
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        if model_name in SCALED_COMPUTE_LOSS:
            self.compute_loss = self.scaled_compute_loss
        if is_training and (not use_eval_mode):
            model.train()
        else:
            model.eval()
        self.validate_model(model, example_inputs)
        return (device, model_name, model, example_inputs, batch_size)

    def iter_model_names(self, args):
        if False:
            while True:
                i = 10
        model_names = sorted(TIMM_MODELS.keys())
        (start, end) = self.get_benchmark_indices(len(model_names))
        for (index, model_name) in enumerate(model_names):
            if index < start or index >= end:
                continue
            if not re.search('|'.join(args.filter), model_name, re.I) or re.search('|'.join(args.exclude), model_name, re.I) or model_name in args.exclude_exact or (model_name in self.skip_models):
                continue
            yield model_name

    def pick_grad(self, name, is_training):
        if False:
            for i in range(10):
                print('nop')
        if is_training:
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        if False:
            for i in range(10):
                print('nop')
        cosine = self.args.cosine
        tolerance = 0.001
        if is_training:
            if REQUIRE_HIGHER_TOLERANCE:
                tolerance = 2 * 0.01
            else:
                tolerance = 0.01
        return (tolerance, cosine)

    def _gen_target(self, batch_size, device):
        if False:
            i = 10
            return i + 15
        return torch.empty((batch_size,) + (), device=device, dtype=torch.long).random_(self.num_classes)

    def compute_loss(self, pred):
        if False:
            for i in range(10):
                print('nop')
        return reduce_to_scalar_loss(pred)

    def scaled_compute_loss(self, pred):
        if False:
            for i in range(10):
                print('nop')
        return reduce_to_scalar_loss(pred) / 1000.0

    def forward_pass(self, mod, inputs, collect_outputs=True):
        if False:
            print('Hello World!')
        with self.autocast():
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        if False:
            for i in range(10):
                print('nop')
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(*cloned_inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None

def timm_main():
    if False:
        print('Hello World!')
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings('ignore')
    main(TimmRunner())
if __name__ == '__main__':
    timm_main()