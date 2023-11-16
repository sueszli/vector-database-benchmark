import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def handler(event, ctx):
    if False:
        return 10
    verification_token = event['verification_token']
    logging.info(f'verification_token={verification_token!r}')
    return {'verification_token': verification_token}