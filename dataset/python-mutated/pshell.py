from . import models

def setup(env):
    if False:
        for i in range(10):
            print('nop')
    request = env['request']
    request.tm.begin()
    env['tm'] = request.tm
    env['dbsession'] = request.dbsession
    env['models'] = models