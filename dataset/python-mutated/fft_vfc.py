from . import fft_python as fft

class fft_vfc:

    def __new__(self, fft_size, forward, window, shift=False, nthreads=1):
        if False:
            return 10
        if forward:
            return fft.fft_vfc_fwd(fft_size, window, shift, nthreads)
        else:
            return fft.fft_vfc_rev(fft_size, window, shift, nthreads)