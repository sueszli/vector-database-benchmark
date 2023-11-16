import numpy as np
import typing

def process_buffer(input_view: cython.int[:, :], output_view: typing.Optional[cython.int[:, :]]=None):
    if False:
        i = 10
        return i + 15
    if output_view is None:
        output_view = np.empty_like(input_view)
    return output_view
process_buffer(None, None)