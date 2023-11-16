import multiprocessing as mp
from multiprocessing.pool import Pool
from stable_diffusion.runner import StableDiffusionRunner
cpu_binding = False
pool = None
var_dict = None

class RunnerProcess(mp.Process):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.runner = StableDiffusionRunner.initialize()

class RunnerPool(Pool):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(1)

    @staticmethod
    def Process(ctx, *args, **kwargs):
        if False:
            while True:
                i = 10
        return RunnerProcess(*args, **kwargs)

def test_func(x):
    if False:
        for i in range(10):
            print('nop')
    print('Running test_func')
    p = mp.current_process()
    y = x * x if p.runner is None else x
    print(y)
if __name__ == '__main__':
    with RunnerPool() as pool:
        for i in range(3):
            print(f'Applying {i} to pool')
            pool.apply(test_func, (i,))