""" An example showcasing the logging system of Sacred."""
import logging
from sacred import Experiment
ex = Experiment('log_example')
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')
ex.logger = logger

@ex.config
def cfg():
    if False:
        return 10
    number = 2
    got_gizmo = False

@ex.capture
def transmogrify(got_gizmo, number, _log):
    if False:
        print('Hello World!')
    if got_gizmo:
        _log.debug('Got gizmo. Performing transmogrification...')
        return number * 42
    else:
        _log.warning("No gizmo. Can't transmogrify!")
        return 0

@ex.automain
def main(number, _log):
    if False:
        i = 10
        return i + 15
    _log.info('Attempting to transmogrify %d...', number)
    result = transmogrify()
    _log.info('Transmogrification complete: %d', result)
    return result