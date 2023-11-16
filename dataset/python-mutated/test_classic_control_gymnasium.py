import subprocess

def test_dqn():
    if False:
        return 10
    subprocess.run('python cleanrl/dqn.py --learning-starts 200 --total-timesteps 205', shell=True, check=True)

def test_c51():
    if False:
        return 10
    subprocess.run('python cleanrl/c51.py --learning-starts 200 --total-timesteps 205', shell=True, check=True)

def test_c51_eval():
    if False:
        print('Hello World!')
    subprocess.run('python cleanrl/c51.py --save-model True --learning-starts 200 --total-timesteps 205', shell=True, check=True)