import numpy as np

def PSNR(data, orig):
    if False:
        while True:
            i = 10
    mse = np.square(np.subtract(data, orig)).mean()
    return 10 * np.log10(255 * 255 / mse)