import numpy as np
__all__ = ['image_stack', 'push', 'pop']
image_stack = []

def push(img):
    if False:
        for i in range(10):
            print('nop')
    'Push an image onto the shared image stack.\n\n    Parameters\n    ----------\n    img : ndarray\n        Image to push.\n\n    '
    if not isinstance(img, np.ndarray):
        raise ValueError('Can only push ndarrays to the image stack.')
    image_stack.append(img)

def pop():
    if False:
        while True:
            i = 10
    'Pop an image from the shared image stack.\n\n    Returns\n    -------\n    img : ndarray\n        Image popped from the stack.\n\n    '
    return image_stack.pop()