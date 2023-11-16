from numba import float64, uint32
from numba.cuda.compiler import compile_ptx
from numba.cuda.testing import skip_on_cudasim, unittest

@skip_on_cudasim('Compilation unsupported in the simulator')
class TestCudaMandel(unittest.TestCase):

    def test_mandel(self):
        if False:
            for i in range(10):
                print('nop')
        'Just make sure we can compile this\n        '

        def mandel(tid, min_x, max_x, min_y, max_y, width, height, iters):
            if False:
                return 10
            pixel_size_x = (max_x - min_x) / width
            pixel_size_y = (max_y - min_y) / height
            x = tid % width
            y = tid / width
            real = min_x + x * pixel_size_x
            imag = min_y + y * pixel_size_y
            c = complex(real, imag)
            z = 0j
            for i in range(iters):
                z = z * z + c
                if z.real * z.real + z.imag * z.imag >= 4:
                    return i
            return iters
        args = (uint32, float64, float64, float64, float64, uint32, uint32, uint32)
        compile_ptx(mandel, args, device=True)
if __name__ == '__main__':
    unittest.main()