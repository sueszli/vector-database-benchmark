import gc
import importlib
import logging
import os
import re
import sys
import warnings
from os.path import abspath, exists
import torch
try:
    from .common import BenchmarkRunner, main
except ImportError:
    from common import BenchmarkRunner, main
from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs
torch.backends.cuda.matmul.allow_tf32 = True

def setup_torchbench_cwd():
    if False:
        i = 10
        return i + 15
    original_dir = abspath(os.getcwd())
    os.environ['KALDI_ROOT'] = '/tmp'
    for torchbench_dir in ('./torchbenchmark', '../torchbenchmark', '../torchbench', '../benchmark', '../../torchbenchmark', '../../torchbench', '../../benchmark'):
        if exists(torchbench_dir):
            break
    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)
    return original_dir
USE_SMALL_BATCH_SIZE = {'demucs': 4, 'dlrm': 1024, 'densenet121': 4, 'hf_Reformer': 4, 'hf_T5_base': 4, 'timm_efficientdet': 1, 'llama_v2_7b_16h': 1, 'yolov3': 8}
INFERENCE_SMALL_BATCH_SIZE = {'timm_efficientdet': 32}
DETECTRON2_MODELS = {'detectron2_fasterrcnn_r_101_c4', 'detectron2_fasterrcnn_r_101_dc5', 'detectron2_fasterrcnn_r_101_fpn', 'detectron2_fasterrcnn_r_50_c4', 'detectron2_fasterrcnn_r_50_dc5', 'detectron2_fasterrcnn_r_50_fpn', 'detectron2_maskrcnn_r_101_c4', 'detectron2_maskrcnn_r_101_fpn', 'detectron2_maskrcnn_r_50_fpn'}
SKIP = {'detectron2_maskrcnn', 'fambench_xlmr', 'tacotron2', 'hf_Bert', 'hf_Bert_large', 'maml', 'clip', 'simple_gpt_tp_manual'}
SKIP_DUE_TO_CONTROL_FLOW = {'cm3leon_generate', 'detectron2_fcos_r_50_fpn', 'fastNLP_Bert', 'hf_Longformer', 'hf_Reformer', 'hf_T5_generate', 'opacus_cifar10', 'speech_transformer'}
SKIP_FOR_CPU = {'hf_T5_generate', 'cm3leon_generate', 'nanogpt', 'sam', 'llama_v2_7b_16h', 'stable_diffusion', 'torchrec_dlrm', 'simple_gpt', 'hf_Whisper', 'stable_diffusion_text_encoder'}
SKIP_FOR_CUDA = {'gat', 'gcn', 'sage'}
SKIP_TRAIN = {'pyhpc_equation_of_state', 'pyhpc_isoneutral_mixing', 'pyhpc_turbulent_kinetic_energy', 'maml', 'llama', 'llama_v2_7b_16h', 'simple_gpt', 'phi_1_5'}
SKIP_TRAIN.update(DETECTRON2_MODELS)
ONLY_TRAINING_MODE = {'tts_angular', 'tacotron2', 'demucs', 'hf_Reformer', 'pytorch_struct', 'yolov3'}
ONLY_TRAINING_MODE.update(DETECTRON2_MODELS)
REQUIRE_HIGHER_TOLERANCE = {'alexnet', 'attention_is_all_you_need_pytorch', 'densenet121', 'hf_Albert', 'vgg16', 'mobilenet_v3_large', 'nvidia_deeprecommender', 'timm_efficientdet'}
REQUIRE_EVEN_HIGHER_TOLERANCE = {'soft_actor_critic', 'tacotron2'}
REQUIRE_HIGHER_FP16_TOLERANCE = {'doctr_reco_predictor', 'drq', 'hf_Whisper'}
REQUIRE_HIGHER_BF16_TOLERANCE = {'doctr_reco_predictor', 'drq', 'hf_Whisper'}
REQUIRE_COSINE_TOLERACE = {}
NONDETERMINISTIC = {'mobilenet_v3_large'}
VERY_SLOW_BENCHMARKS = {'hf_BigBird', 'hf_Longformer', 'hf_T5'}
SLOW_BENCHMARKS = {*VERY_SLOW_BENCHMARKS, 'BERT_pytorch', 'demucs', 'fastNLP_Bert', 'hf_Albert', 'hf_Bart', 'hf_Bert', 'hf_DistilBert', 'hf_GPT2', 'hf_Reformer', 'speech_transformer', 'vision_maskrcnn'}
TRT_NOT_YET_WORKING = {'alexnet', 'resnet18', 'resnet50', 'mobilenet_v2', 'mnasnet1_0', 'squeezenet1_1', 'shufflenetv2_x1_0', 'vgg16', 'resnext50_32x4d'}
DONT_CHANGE_BATCH_SIZE = {'demucs', 'pytorch_struct', 'pyhpc_turbulent_kinetic_energy', 'vision_maskrcnn'}
SKIP_ACCURACY_CHECK_MODELS = {'hf_GPT2_large', 'hf_T5_large', 'timm_vision_transformer_large', 'maml', 'llama_v2_7b_16h', 'Background_Matting', 'stable_diffusion_unet'}
SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS = {'Background_Matting'}
MAX_BATCH_SIZE_FOR_ACCURACY_CHECK = {'hf_GPT2': 2, 'pytorch_unet': 2}
FORCE_AMP_FOR_FP16_BF16_MODELS = {'DALLE2_pytorch', 'doctr_det_predictor', 'doctr_reco_predictor', 'Super_SloMo', 'tts_angular', 'pyhpc_turbulent_kinetic_energy', 'detectron2_fcos_r_50_fpn'}
FORCE_FP16_FOR_BF16_MODELS = {'vision_maskrcnn'}
CANARY_MODELS = {'torchrec_dlrm', 'clip'}
ONLY_MULTIPROCESS = {'simple_gpt'}

class TorchBenchmarkRunner(BenchmarkRunner):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.suite_name = 'torchbench'
        self.optimizer = None

    @property
    def skip_models(self):
        if False:
            for i in range(10):
                print('nop')
        return SKIP

    @property
    def skip_models_for_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        return SKIP_FOR_CPU

    @property
    def skip_models_for_cuda(self):
        if False:
            while True:
                i = 10
        return SKIP_FOR_CUDA

    @property
    def slow_models(self):
        if False:
            while True:
                i = 10
        return SLOW_BENCHMARKS

    @property
    def very_slow_models(self):
        if False:
            for i in range(10):
                print('nop')
        return VERY_SLOW_BENCHMARKS

    @property
    def non_deterministic_models(self):
        if False:
            return 10
        return NONDETERMINISTIC

    @property
    def skip_not_suitable_for_training_models(self):
        if False:
            i = 10
            return i + 15
        return SKIP_TRAIN

    @property
    def failing_fx2trt_models(self):
        if False:
            return 10
        return TRT_NOT_YET_WORKING

    @property
    def force_amp_for_fp16_bf16_models(self):
        if False:
            i = 10
            return i + 15
        return FORCE_AMP_FOR_FP16_BF16_MODELS

    @property
    def force_fp16_for_bf16_models(self):
        if False:
            return 10
        return FORCE_FP16_FOR_BF16_MODELS

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if False:
            print('Hello World!')
        if self.args.dashboard or self.args.accuracy:
            return SKIP_ACCURACY_CHECK_MODELS
        return set()

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        if False:
            print('Hello World!')
        if self.args.accuracy and self.args.training:
            return SKIP_ACCURACY_CHECK_AS_EAGER_NON_DETERMINISTIC_MODELS
        return set()

    @property
    def skip_multiprocess_models(self):
        if False:
            return 10
        return ONLY_MULTIPROCESS

    @property
    def skip_models_due_to_control_flow(self):
        if False:
            i = 10
            return i + 15
        return SKIP_DUE_TO_CONTROL_FLOW

    def load_model(self, device, model_name, batch_size=None, part=None, extra_args=None):
        if False:
            for i in range(10):
                print('nop')
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError('Activation checkpointing not implemented for Torchbench models')
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dynamic_shapes = self.args.dynamic_shapes
        candidates = [f'torchbenchmark.models.{model_name}', f'torchbenchmark.canary_models.{model_name}', f'torchbenchmark.models.fb.{model_name}']
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError as e:
                if e.name != c:
                    raise
        else:
            raise ImportError(f'could not import any of {candidates}')
        benchmark_cls = getattr(module, 'Model', None)
        if not hasattr(benchmark_cls, 'name'):
            benchmark_cls.name = model_name
        cant_change_batch_size = not getattr(benchmark_cls, 'ALLOW_CUSTOMIZE_BSIZE', True) or model_name in DONT_CHANGE_BATCH_SIZE
        if cant_change_batch_size:
            batch_size = None
        if batch_size is None and is_training and (model_name in USE_SMALL_BATCH_SIZE):
            batch_size = USE_SMALL_BATCH_SIZE[model_name]
        elif batch_size is None and (not is_training) and (model_name in INFERENCE_SMALL_BATCH_SIZE):
            batch_size = INFERENCE_SMALL_BATCH_SIZE[model_name]
        if self.args.accuracy and model_name in MAX_BATCH_SIZE_FOR_ACCURACY_CHECK:
            batch_size = min(batch_size, MAX_BATCH_SIZE_FOR_ACCURACY_CHECK[model_name])
        torch.backends.__allow_nonbracketed_mutation_flag = True
        if extra_args is None:
            extra_args = []
        if part:
            extra_args += ['--part', part]
        if model_name == 'vision_maskrcnn' and is_training:
            model_kwargs = {'box_detections_per_img': 5}
            benchmark = benchmark_cls(test='train', device=device, batch_size=batch_size, extra_args=extra_args, model_kwargs=model_kwargs)
        elif is_training:
            benchmark = benchmark_cls(test='train', device=device, batch_size=batch_size, extra_args=extra_args)
        else:
            benchmark = benchmark_cls(test='eval', device=device, batch_size=batch_size, extra_args=extra_args)
        (model, example_inputs) = benchmark.get_module()
        if is_training and (not use_eval_mode or model_name in ONLY_TRAINING_MODE):
            model.train()
        else:
            model.eval()
        gc.collect()
        batch_size = benchmark.batch_size
        if model_name == 'yolov3':
            example_inputs = (torch.rand(batch_size, 3, 384, 512).to(device),)
        if model_name == 'maml_omniglot':
            batch_size = 5
            assert example_inputs[0].shape[0] == batch_size
        if model_name == 'vision_maskrcnn':
            batch_size = 1
        if self.args.trace_on_xla:
            import torch_xla
        self.validate_model(model, example_inputs)
        return (device, benchmark.name, model, example_inputs, batch_size)

    def iter_model_names(self, args):
        if False:
            while True:
                i = 10
        from torchbenchmark import _list_canary_model_paths, _list_model_paths
        models = _list_model_paths()
        models += [f for f in _list_canary_model_paths() if os.path.basename(f) in CANARY_MODELS]
        models.sort()
        (start, end) = self.get_benchmark_indices(len(models))
        for (index, model_path) in enumerate(models):
            if index < start or index >= end:
                continue
            model_name = os.path.basename(model_path)
            if not re.search('|'.join(args.filter), model_name, re.I) or re.search('|'.join(args.exclude), model_name, re.I) or model_name in args.exclude_exact or (model_name in self.skip_models):
                continue
            yield model_name

    def pick_grad(self, name, is_training):
        if False:
            i = 10
            return i + 15
        if is_training or name in ('maml',):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        if False:
            for i in range(10):
                print('nop')
        tolerance = 0.0001
        cosine = self.args.cosine
        if self.args.float16 or self.args.amp:
            if name in REQUIRE_HIGHER_FP16_TOLERANCE:
                return (0.01, cosine)
            return (0.001, cosine)
        if self.args.bfloat16:
            if name in REQUIRE_HIGHER_BF16_TOLERANCE:
                return (0.01, cosine)
        if is_training and current_device == 'cuda':
            tolerance = 0.001
            if name in REQUIRE_COSINE_TOLERACE:
                cosine = True
            elif name in REQUIRE_HIGHER_TOLERANCE:
                tolerance = 0.001
            elif name in REQUIRE_EVEN_HIGHER_TOLERANCE:
                tolerance = 8 * 0.01
        return (tolerance, cosine)

    def compute_loss(self, pred):
        if False:
            return 10
        return reduce_to_scalar_loss(pred)

    def forward_pass(self, mod, inputs, collect_outputs=True):
        if False:
            return 10
        with self.autocast():
            return mod(*inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        if False:
            return 10
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast():
            pred = mod(*cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None

def torchbench_main():
    if False:
        for i in range(10):
            print('nop')
    original_dir = setup_torchbench_cwd()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings('ignore')
    main(TorchBenchmarkRunner(), original_dir)
if __name__ == '__main__':
    torchbench_main()