import importlib
import logging
import types
_logger = logging.getLogger(__name__)
SUPPORTED_DEBUGGER = {'pdb', 'ipdb', 'wdb', 'pudb'}

def post_mortem(config, info):
    if False:
        i = 10
        return i + 15
    if config['dev_mode'] and isinstance(info[2], types.TracebackType):
        debug = next((opt for opt in config['dev_mode'] if opt in SUPPORTED_DEBUGGER), None)
        if debug:
            try:
                importlib.import_module(debug).post_mortem(info[2])
            except ImportError:
                _logger.error('Error while importing %s.' % debug)