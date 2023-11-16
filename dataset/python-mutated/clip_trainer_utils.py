import math
import os
from functools import partial
from inspect import unwrap
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from modelscope.outputs import OutputKeys

def get_optimizer_params(model_name, cfg):
    if False:
        return 10
    if model_name in ['damo/multi-modal_clip-vit-base-patch16_zh']:
        params = {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-06, 'weight_decay': 0.0}
    elif model_name in ['damo/multi-modal_clip-vit-large-patch14_zh', 'damo/multi-modal_clip-vit-large-patch14_336_zh']:
        params = {'lr': 0.0004, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-06, 'weight_decay': 0.0}
    else:
        params = {'lr': 0.0005, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-08, 'weight_decay': 0.0}
    for key in ['lr', 'beta1', 'beta2', 'eps', 'weight_decay']:
        if hasattr(cfg.train, 'optimizer_hparams'):
            params[key] = getattr(cfg.train.optimizer_hparams, key, params[key])
    return params

def get_loss(model_outputs, loss_img, loss_txt, loss_cfg):
    if False:
        print('Hello World!')
    image_features = model_outputs[OutputKeys.IMG_EMBEDDING]
    text_features = model_outputs[OutputKeys.TEXT_EMBEDDING]
    logit_scale = model_outputs['logit_scale']
    logit_scale = logit_scale.mean()
    if loss_cfg.aggregate and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        all_image_features = torch.cat([image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1:])
        all_text_features = torch.cat([text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1:])
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(int(os.environ.get('LOCAL_RANK', 0)), non_blocking=True)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    return total_loss

def lr_lambda(num_warmup_steps, num_training_steps, num_cycles, current_step):
    if False:
        return 10
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_schedule(optimizer, scheduler, num_cycles: float=0.5, last_epoch: int=-1):
    if False:
        while True:
            i = 10
    num_warmup_steps = int(scheduler.warmup_proportion * scheduler.num_train_steps)
    num_training_steps = scheduler.num_train_steps
    return LambdaLR(optimizer, partial(lr_lambda, num_warmup_steps, num_training_steps, num_cycles), last_epoch)