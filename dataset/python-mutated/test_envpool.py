import subprocess

def test_ppo_atari_envpool():
    if False:
        return 10
    subprocess.run('python cleanrl/ppo_atari_envpool.py --num-envs 8 --num-steps 32 --total-timesteps 256', shell=True, check=True)

def test_ppo_rnd_envpool():
    if False:
        return 10
    subprocess.run('python cleanrl/ppo_rnd_envpool.py --num-envs 8 --num-steps 32 --num-iterations-obs-norm-init 1 --total-timesteps 256', shell=True, check=True)

def test_ppo_atari_envpool_xla_jax():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/ppo_atari_envpool_xla_jax.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256', shell=True, check=True)

def test_ppo_atari_envpool_xla_jax_scan():
    if False:
        for i in range(10):
            print('nop')
    subprocess.run('python cleanrl/ppo_atari_envpool_xla_jax_scan.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256', shell=True, check=True)

def test_ppo_atari_envpool_xla_jax_scan_eval():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/ppo_atari_envpool_xla_jax_scan.py --save-model True --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256', shell=True, check=True)