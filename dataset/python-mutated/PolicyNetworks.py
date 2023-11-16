import torch
policy_arch_dict = {'None': None, 'evo_ppo2.1': dict(activation_fn=torch.nn.ReLU, net_arch=[256, 128, dict(vf=[128, 64], pi=[64, 32])]), 'drone_ppo': dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]), 'drone_sac': dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, 256, 128])}

def get_policy_arch(arch_str):
    if False:
        for i in range(10):
            print('nop')
    return policy_arch_dict.get(arch_str, None)