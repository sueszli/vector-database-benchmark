import subprocess

def test_dqn_jax():
    if False:
        while True:
            i = 10
    subprocess.run('python cleanrl/dqn_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_dqn_jax_eval():
    if False:
        for i in range(10):
            print('nop')
    subprocess.run('python cleanrl/dqn_atari_jax.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_qdagger_dqn_atari_jax_impalacnn():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1', shell=True, check=True)

def test_qdagger_dqn_atari_jax_impalacnn_eval():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1', shell=True, check=True)

def test_c51_atari_jax():
    if False:
        print('Hello World!')
    subprocess.run('python cleanrl/c51_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_c51_atari_jax_eval():
    if False:
        for i in range(10):
            print('nop')
    subprocess.run('python cleanrl/c51_atari_jax.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)