import subprocess

def test_dqn_jax():
    if False:
        while True:
            i = 10
    subprocess.run('python cleanrl/dqn_jax.py --learning-starts 200 --total-timesteps 205', shell=True, check=True)

def test_c51_jax():
    if False:
        return 10
    subprocess.run('python cleanrl/c51_jax.py --learning-starts 200 --total-timesteps 205', shell=True, check=True)

def test_c51_jax_eval():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/c51_jax.py --save-model True --learning-starts 200 --total-timesteps 205', shell=True, check=True)