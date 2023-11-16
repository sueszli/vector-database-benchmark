import subprocess

def test_dqn():
    if False:
        while True:
            i = 10
    subprocess.run('python enjoy.py --exp-name dqn --env CartPole-v1 --eval-episodes 1', shell=True, check=True)

def test_dqn_atari():
    if False:
        print('Hello World!')
    subprocess.run('python enjoy.py --exp-name dqn_atari --env BreakoutNoFrameskip-v4 --eval-episodes 1', shell=True, check=True)

def test_dqn_jax():
    if False:
        return 10
    subprocess.run('python enjoy.py --exp-name dqn_jax --env CartPole-v1 --eval-episodes 1', shell=True, check=True)