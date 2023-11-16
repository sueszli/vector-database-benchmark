import torch
from torch.multiprocessing import Process
import nvidia.dali as dali

def task_function():
    if False:
        i = 10
        return i + 15
    torch.cuda.set_device(0)

def test_actual_proc():
    if False:
        print('Hello World!')
    phase_process = Process(target=task_function)
    phase_process.start()
    phase_process.join()
    assert phase_process.exitcode == 0