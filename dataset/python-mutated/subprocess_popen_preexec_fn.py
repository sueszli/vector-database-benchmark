import subprocess

def foo():
    if False:
        while True:
            i = 10
    pass
subprocess.Popen(preexec_fn=foo)
subprocess.Popen(['ls'], preexec_fn=foo)
subprocess.Popen(preexec_fn=lambda : print('Hello, world!'))
subprocess.Popen(['ls'], preexec_fn=lambda : print('Hello, world!'))
subprocess.Popen()
subprocess.Popen(['ls'])
subprocess.Popen(preexec_fn=None)
subprocess.Popen(['ls'], preexec_fn=None)