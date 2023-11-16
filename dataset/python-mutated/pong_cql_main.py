import torch
from copy import deepcopy
from ding.entry import serial_pipeline_offline, collect_demo_data, eval, serial_pipeline

def train_cql(args):
    if False:
        for i in range(10):
            print('nop')
    from dizoo.atari.config.serial.pong.pong_cql_config import main_config, create_config
    main_config.exp_name = 'pong_cql'
    main_config.policy.collect.data_path = './pong/expert_demos.hdf5'
    main_config.policy.collect.data_type = 'hdf5'
    config = deepcopy([main_config, create_config])
    serial_pipeline_offline(config, seed=args.seed)

def eval_ckpt(args):
    if False:
        for i in range(10):
            print('nop')
    from dizoo.atari.config.serial.pong.pong_qrdqn_generation_data_config import main_config, create_config
    main_config.exp_name = 'pong'
    config = deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path='./pong/ckpt/ckpt_best.pth.tar')

def generate(args):
    if False:
        i = 10
        return i + 15
    from dizoo.atari.config.serial.pong.pong_qrdqn_generation_data_config import main_config, create_config
    main_config.exp_name = 'pong'
    main_config.policy.collect.save_path = './pong/expert.pkl'
    config = deepcopy([main_config, create_config])
    state_dict = torch.load('./pong/ckpt/ckpt_best.pth.tar', map_location='cpu')
    collect_demo_data(config, collect_count=int(100000.0), seed=args.seed, expert_data_path=main_config.policy.collect.save_path, state_dict=state_dict)

def train_expert(args):
    if False:
        print('Hello World!')
    from dizoo.atari.config.serial.pong.pong_qrdqn_config import main_config, create_config
    main_config.exp_name = 'pong'
    config = deepcopy([main_config, create_config])
    serial_pipeline(config, seed=args.seed, max_iterations=1000000.0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    args = parser.parse_args()
    train_expert(args)
    eval_ckpt(args)
    generate(args)
    train_cql(args)