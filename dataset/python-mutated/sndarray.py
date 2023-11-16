"""pygame module for accessing sound sample data

Functions to convert between NumPy arrays and Sound objects. This module
will only be functional when pygame can use the external NumPy package.
If NumPy can't be imported, surfarray becomes a MissingModule object.

Sound data is made of thousands of samples per second, and each sample
is the amplitude of the wave at a particular moment in time. For
example, in 22-kHz format, element number 5 of the array is the
amplitude of the wave after 5/22000 seconds.

Each sample is an 8-bit or 16-bit integer, depending on the data format.
A stereo sound file has two values per sample, while a mono sound file
only has one.

Sounds with 16-bit data will be treated as unsigned integers,
if the sound sample type requests this.
"""
from pygame import mixer
import numpy
import warnings
__all__ = ['array', 'samples', 'make_sound', 'use_arraytype', 'get_arraytype', 'get_arraytypes']

def array(sound):
    if False:
        while True:
            i = 10
    'pygame.sndarray.array(Sound): return array\n\n    Copy Sound samples into an array.\n\n    Creates a new array for the sound data and copies the samples. The\n    array will always be in the format returned from\n    pygame.mixer.get_init().\n    '
    return numpy.array(sound, copy=True)

def samples(sound):
    if False:
        for i in range(10):
            print('nop')
    'pygame.sndarray.samples(Sound): return array\n\n    Reference Sound samples into an array.\n\n    Creates a new array that directly references the samples in a Sound\n    object. Modifying the array will change the Sound. The array will\n    always be in the format returned from pygame.mixer.get_init().\n    '
    return numpy.array(sound, copy=False)

def make_sound(array):
    if False:
        i = 10
        return i + 15
    'pygame.sndarray.make_sound(array): return Sound\n\n    Convert an array into a Sound object.\n\n    Create a new playable Sound object from an array. The mixer module\n    must be initialized and the array format must be similar to the mixer\n    audio format.\n    '
    return mixer.Sound(array=array)

def use_arraytype(arraytype):
    if False:
        return 10
    'pygame.sndarray.use_arraytype(arraytype): return None\n\n    DEPRECATED - only numpy arrays are now supported.\n    '
    warnings.warn(DeprecationWarning('only numpy arrays are now supported, this function will be removed in a future version of the module'))
    arraytype = arraytype.lower()
    if arraytype != 'numpy':
        raise ValueError('invalid array type')

def get_arraytype():
    if False:
        return 10
    'pygame.sndarray.get_arraytype(): return str\n\n    DEPRECATED - only numpy arrays are now supported.\n    '
    warnings.warn(DeprecationWarning('only numpy arrays are now supported, this function will be removed in a future version of the module'))
    return 'numpy'

def get_arraytypes():
    if False:
        print('Hello World!')
    'pygame.sndarray.get_arraytypes(): return tuple\n\n    DEPRECATED - only numpy arrays are now supported.\n    '
    warnings.warn(DeprecationWarning('only numpy arrays are now supported, this function will be removed in a future version of the module'))
    return ('numpy',)