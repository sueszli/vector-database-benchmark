import logging
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    log.info('master tops test loaded')
    return 'master_tops_test'

def top(**kwargs):
    if False:
        print('Hello World!')
    log.info('master_tops_test')
    return {'base': ['master_tops_test']}