from robot.api import logger
from robot.api.deco import keyword

@keyword(name='${a}*lib*${b}')
def mult_match3(a, b):
    if False:
        while True:
            i = 10
    logger.info('%s*lib*%s' % (a, b))