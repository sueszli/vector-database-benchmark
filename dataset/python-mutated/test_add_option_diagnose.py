from loguru import logger

def test_diagnose(writer):
    if False:
        return 10
    logger.add(writer, format='{message}', diagnose=True)
    try:
        1 / 0
    except Exception:
        logger.exception('')
    result_with = writer.read().strip()
    logger.remove()
    writer.clear()
    logger.add(writer, format='{message}', diagnose=False)
    try:
        1 / 0
    except Exception:
        logger.exception('')
    result_without = writer.read().strip()
    assert len(result_with.splitlines()) > len(result_without.splitlines())