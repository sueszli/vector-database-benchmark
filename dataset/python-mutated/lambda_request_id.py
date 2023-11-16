import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def handler(event, context):
    if False:
        return 10
    LOGGER.info('RequestId log message')
    return context.aws_request_id