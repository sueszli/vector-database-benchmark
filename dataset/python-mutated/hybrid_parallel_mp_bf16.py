import unittest
from hybrid_parallel_mp_model import TestDistMPTraining
import paddle
from paddle.distributed import fleet
from paddle.distributed.utils.nccl_utils import check_nccl_version_for_bf16

class TestMPFP16(TestDistMPTraining):

    def build_optimizer(self, model):
        if False:
            while True:
                i = 10
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.999, verbose=True)
        optimizer = paddle.optimizer.SGD(scheduler, grad_clip=grad_clip, parameters=model.parameters())
        (model, optimizer) = paddle.amp.decorate(models=model, optimizers=optimizer, dtype='bfloat16', level='O2', save_dtype='float32')
        return optimizer

    def train_batch(self, batch, model, optimizer, is_mp):
        if False:
            i = 10
            return i + 15
        scaler = paddle.amp.GradScaler(init_loss_scaling=1, use_dynamic_loss_scaling=False)
        if is_mp:
            scaler = fleet.distributed_scaler(scaler)
        with paddle.amp.auto_cast(enable=True, dtype='bfloat16', level='O2'):
            output = model(batch)
            loss = output.mean()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()
        return scaled
if __name__ == '__main__':
    if check_nccl_version_for_bf16() and paddle.device.cuda.get_device_properties().major >= 8:
        unittest.main()