from loguru import logger

def test_backtrace(writer):
    if False:
        print('Hello World!')
    logger.add(writer, format='{message}', backtrace=True)
    try:
        1 / 0
    except Exception:
        logger.exception('')
    result_with = writer.read().strip()
    logger.remove()
    writer.clear()
    logger.add(writer, format='{message}', backtrace=False)
    try:
        1 / 0
    except Exception:
        logger.exception('')
    result_without = writer.read().strip()
    assert len(result_with.splitlines()) > len(result_without.splitlines())