import logging
import signal
from typing import Any, Dict
logger = logging.getLogger(__name__)

def start_trading(args: Dict[str, Any]) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Main entry point for trading mode\n    '
    from freqtrade.worker import Worker

    def term_handler(signum, frame):
        if False:
            while True:
                i = 10
        raise KeyboardInterrupt()
    worker = None
    try:
        signal.signal(signal.SIGTERM, term_handler)
        worker = Worker(args)
        worker.run()
    except Exception as e:
        logger.error(str(e))
        logger.exception('Fatal exception!')
    except KeyboardInterrupt:
        logger.info('SIGINT received, aborting ...')
    finally:
        if worker:
            logger.info('worker found ... calling exit')
            worker.exit()
    return 0