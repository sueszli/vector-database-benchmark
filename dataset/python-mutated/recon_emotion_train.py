import argparse
import collections
import torch
import numpy as np
import recon_data_generator
import recon_losses_R
import recon_metrics_R
import reconmodels as module_arch
from parse_config import ConfigParser
from recon_trainer import Trainer
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    if False:
        while True:
            i = 10
    logger = config.get_logger('train')
    data_loader = config.init_obj('recon_data_generator', recon_data_generator)
    valid_data_loader = data_loader.split_validation()
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    criterion = getattr(recon_losses_R, config['loss'])
    metrics = [getattr(recon_metrics_R, met) for met in config['metrics']]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer, config=config, data_loader=data_loader, valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler)
    trainer.train()
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'), CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')]
    config = ConfigParser.from_args(args, options)