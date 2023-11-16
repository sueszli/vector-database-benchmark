from typing import List
import torch
from torch import nn
from modelscope.utils.logger import get_logger
logger = get_logger()

def create_feature_extractor(model_type, **kwargs):
    if False:
        while True:
            i = 10
    ' Create the feature extractor for <model_type> architecture. '
    if model_type == 'ddpm':
        logger.info('Creating DDPM Feature Extractor...')
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f'Wrong model type: {model_type}')
    return feature_extractor

def save_tensors(module: nn.Module, features, name: str):
    if False:
        for i in range(10):
            print('nop')
    ' Process and save activations in the module. '
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for (k, f) in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    if False:
        return 10
    save_tensors(self, out, 'activations')

def save_input_hook(self, inp, out):
    if False:
        return 10
    save_tensors(self, inp[0], 'activations')

class FeatureExtractor(nn.Module):

    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Parent feature extractor class.\n\n        param: model_path: path to the pretrained model\n        param: input_activations:\n            If True, features are input activations of the corresponding blocks\n            If False, features are output activations of the corresponding blocks\n        '
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        logger.info(f'Pretrained model is successfully loaded from {model_path}')
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

class FeatureExtractorDDPM(FeatureExtractor):
    """
    Wrapper to extract features from pretrained DDPMs.

    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    """

    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.steps = steps
        for (idx, block) in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        if False:
            print('Hello World!')
        import inspect
        from ddpm_guided_diffusion.script_util import create_model_and_diffusion
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        (self.model, self.diffusion) = create_model_and_diffusion(**expected_args)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        if False:
            for i in range(10):
                print('nop')
        activations = []
        for t in self.steps:
            t = torch.tensor([t]).to(x.device)
            noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            self.model(noisy_x, self.diffusion._scale_timesteps(t))
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None
        return activations

def collect_features(cfg, activations: List[torch.Tensor], sample_idx=0):
    if False:
        for i in range(10):
            print('nop')
    ' Upsample activations and concatenate them to form a feature tensor '
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(cfg.mlp.dim[:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(feats, size=size, mode=cfg.ddpm.upsample_mode)
        resized_activations.append(feats[0])
    return torch.cat(resized_activations, dim=0)