import pytest
import celery

def test_version():
    if False:
        print('Hello World!')
    assert celery.VERSION
    assert len(celery.VERSION) >= 3
    celery.VERSION = (0, 3, 0)
    assert celery.__version__.count('.') >= 2

@pytest.mark.parametrize('attr', ['__author__', '__contact__', '__homepage__', '__docformat__'])
def test_meta(attr):
    if False:
        print('Hello World!')
    assert getattr(celery, attr, None)