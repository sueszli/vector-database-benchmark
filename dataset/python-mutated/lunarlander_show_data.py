from dizoo.classic_control.cartpole.offline_data.collect_dqn_data_config import main_config, create_config
from ding.entry import serial_pipeline_offline
import os
import torch
from torch.utils.data import DataLoader
from ding.config import read_config, compile_config
from ding.utils.data import create_dataset

def train(args):
    if False:
        while True:
            i = 10
    config = [main_config, create_config]
    input_cfg = config
    if isinstance(input_cfg, str):
        (cfg, create_cfg) = read_config(input_cfg)
    else:
        (cfg, create_cfg) = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=args.seed, auto=True, create_cfg=create_cfg)
    dataset = create_dataset(cfg)
    print(dataset.__len__())
    print(dataset.__getitem__(0)[0]['action'])
    episode_action = torch.stack([dataset.__getitem__(0)[i]['action'] for i in range(dataset.__getitem__(0).__len__())], axis=0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    train(args)