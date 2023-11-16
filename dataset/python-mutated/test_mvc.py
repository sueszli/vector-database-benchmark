import multiprocessing as mp
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_under_cuda_memcheck, skip_if_mvc_libraries_unavailable
from numba.tests.support import linux_only

def child_test():
    if False:
        i = 10
        return i + 15
    from numba import config, cuda
    config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1

    @cuda.jit
    def f():
        if False:
            for i in range(10):
                print('nop')
        pass
    f[1, 1]()

def child_test_wrapper(result_queue):
    if False:
        while True:
            i = 10
    try:
        output = child_test()
        success = True
    except:
        output = traceback.format_exc()
        success = False
    result_queue.put((success, output))

@linux_only
@skip_under_cuda_memcheck('May hang CUDA memcheck')
@skip_on_cudasim('Simulator does not require or implement MVC')
@skip_if_mvc_libraries_unavailable
class TestMinorVersionCompatibility(CUDATestCase):

    def test_mvc(self):
        if False:
            while True:
                i = 10
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        proc = ctx.Process(target=child_test_wrapper, args=(result_queue,))
        proc.start()
        proc.join()
        (success, output) = result_queue.get()
        if not success:
            self.fail(output)
if __name__ == '__main__':
    unittest.main()