import logging
import numpy as np
logger = logging.getLogger(__name__)
try:
    from numba import njit
except (ImportError, ModuleNotFoundError):
    logger.debug("Numba not found, replacing njit() with no-op implementation. Enable it with 'pip install numba'.")

    def njit(f):
        if False:
            while True:
                i = 10
        return f

@njit
def expit(x: float) -> float:
    if False:
        while True:
            i = 10
    return 1 / (1 + np.exp(-x))