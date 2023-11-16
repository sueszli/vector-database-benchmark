import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

def app_factory(global_options, **local_options):
    if False:
        print('Hello World!')
    return app

def app(environ, start_response):
    if False:
        while True:
            i = 10
    start_response('200 OK', [])
    log.debug('Hello Debug!')
    log.info('Hello Info!')
    log.warn('Hello Warn!')
    log.error('Hello Error!')
    return [b'Hello World!\n']