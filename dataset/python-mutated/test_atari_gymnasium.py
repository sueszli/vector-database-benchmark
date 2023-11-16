import subprocess

def test_dqn():
    if False:
        print('Hello World!')
    subprocess.run('python cleanrl/dqn_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_dqn_eval():
    if False:
        print('Hello World!')
    subprocess.run('python cleanrl/dqn_atari.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_qdagger_dqn_atari_impalacnn():
    if False:
        return 10
    subprocess.run('python cleanrl/qdagger_dqn_atari_impalacnn.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1', shell=True, check=True)

def test_qdagger_dqn_atari_impalacnn_eval():
    if False:
        for i in range(10):
            print('nop')
    subprocess.run('python cleanrl/qdagger_dqn_atari_impalacnn.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1', shell=True, check=True)

def test_c51_atari():
    if False:
        print('Hello World!')
    subprocess.run('python cleanrl/c51_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_c51_atari_eval():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/c51_atari.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)

def test_sac():
    if False:
        i = 10
        return i + 15
    subprocess.run('python cleanrl/sac_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4', shell=True, check=True)