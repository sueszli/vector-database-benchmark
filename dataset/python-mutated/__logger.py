"""
@version: 0.01
@brief: 日志模块
"""
import logging
from logging.handlers import RotatingFileHandler
import os
import re
tarsLogger = logging.getLogger('TARS client')
strToLoggingLevel = {'critical': logging.CRITICAL, 'error': logging.ERROR, 'warn': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG, 'none': logging.NOTSET}

def createLogFile(filename):
    if False:
        while True:
            i = 10
    if filename.endswith('/'):
        raise ValueError('The logfile is a dir not a file')
    if os.path.exists(filename) and os.path.isfile(filename):
        pass
    else:
        fileComposition = str.split(filename, '/')
        print(fileComposition)
        currentFile = ''
        for item in fileComposition:
            if item == fileComposition[-1]:
                currentFile += item
                if not os.path.exists(currentFile) or not os.path.isfile(currentFile):
                    while True:
                        try:
                            os.mknod(currentFile)
                            break
                        except OSError as msg:
                            errno = re.findall('\\d+', str(msg))
                            if len(errno) > 0 and errno[0] == '17':
                                currentFile += '.log'
                                continue
                break
            currentFile += item + '/'
            if not os.path.exists(currentFile):
                os.mkdir(currentFile)

def initLog(logpath, logsize, lognum, loglevel):
    if False:
        i = 10
        return i + 15
    createLogFile(logpath)
    handler = RotatingFileHandler(filename=logpath, maxBytes=logsize, backupCount=lognum)
    formatter = logging.Formatter('%(asctime)s | %(levelname)6s | [%(filename)18s:%(lineno)4d] | [%(thread)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    tarsLogger.addHandler(handler)
    if loglevel in strToLoggingLevel:
        tarsLogger.setLevel(strToLoggingLevel[loglevel])
    else:
        tarsLogger.setLevel(strToLoggingLevel['error'])
if __name__ == '__main__':
    tarsLogger.debug('debug log')
    tarsLogger.info('info log')
    tarsLogger.warning('warning log')
    tarsLogger.error('error log')
    tarsLogger.critical('critical log')