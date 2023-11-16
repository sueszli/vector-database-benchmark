from . import fft_python as fft

class fft_vcc:

    def __new__(self, fft_size, forward, window, shift=False, nthreads=1):
        if False:
            for i in range(10):
                print('nop')
        if forward:
            return fft.fft_vcc_fwd(fft_size, window, shift, nthreads)
        else:
            return fft.fft_vcc_rev(fft_size, window, shift, nthreads)