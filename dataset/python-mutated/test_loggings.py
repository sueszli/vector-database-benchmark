import logging

def test_basic_logging(caplog):
    if False:
        for i in range(10):
            print('nop')
    caplog.set_level(logging.INFO)
    logging.info('foo')
    assert 'foo' in caplog.text