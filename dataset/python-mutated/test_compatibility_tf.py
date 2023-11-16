import pytest

@pytest.mark.gpu
def test_compatibility_tf():
    if False:
        return 10
    'Some of our code uses TF1 and some TF2. Here we just check that we\n    can import both versions.\n    '
    import tensorflow as tf
    from tensorflow.compat.v1 import placeholder