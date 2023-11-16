import pytest
import requests
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    import torch
except ImportError:
    pass

def test_requests():
    if False:
        while True:
            i = 10
    assert requests.__version__ >= '2.31.0'

def test_numpy():
    if False:
        print('Hello World!')
    assert np.__version__ >= '1.13.3'

def test_pandas():
    if False:
        print('Hello World!')
    assert pd.__version__ >= '1.0.3'

@pytest.mark.gpu
def test_tensorflow():
    if False:
        i = 10
        return i + 15
    assert tf.__version__ >= '2.8.4'

@pytest.mark.gpu
def test_torch():
    if False:
        for i in range(10):
            print('nop')
    assert torch.__version__ >= '1.13.1'