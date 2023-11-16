import logging
logger = logging.getLogger('pupy')

def getLogger(name):
    if False:
        return 10
    return logger.getChild(name)