import os
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from tests_pytorch.helpers.runif import RunIf

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_collate_checkpoint(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test to ensure that with DeepSpeed Stage 3 we can collate the sharded checkpoints into a single file.'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, strategy=DeepSpeedStrategy(stage=3), accelerator='gpu', devices=2, fast_dev_run=True, precision='16-mixed', enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model)
    checkpoint_path = os.path.join(tmpdir, 'model.pt')
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    if trainer.is_global_zero:
        output_path = os.path.join(tmpdir, 'single_model.pt')
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)
        _assert_checkpoint_equal(model, output_path)

def _assert_checkpoint_equal(model, output_path):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.exists(output_path)
    single_output = torch.load(output_path)
    state_dict = model.state_dict()
    for (orig_param, saved_model_param) in zip(state_dict.values(), single_output['state_dict'].values()):
        if model.dtype == torch.half:
            saved_model_param = saved_model_param.half()
        assert torch.equal(orig_param.cpu(), saved_model_param)