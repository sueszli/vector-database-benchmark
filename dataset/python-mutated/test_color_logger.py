import logging
from kedro.extras.logging import ColorHandler

def test_color_logger(caplog):
    if False:
        return 10
    log = logging.getLogger(__name__)
    for handler in log.handlers:
        log.removeHandler(handler)
    log.addHandler(ColorHandler())
    log.info('Test')
    for record in caplog.records:
        assert record.levelname == 'INFO'
        assert 'Test' in record.msg