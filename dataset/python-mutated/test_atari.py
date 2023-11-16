import subprocess

def test_ppo():
    if False:
        for i in range(10):
            print('nop')
    subprocess.run('python cleanrl/ppo_atari.py --num-envs 1 --num-steps 64 --total-timesteps 256', shell=True, check=True)

def test_ppo_lstm():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/ppo_atari_lstm.py --num-envs 4 --num-steps 64 --total-timesteps 256', shell=True, check=True)