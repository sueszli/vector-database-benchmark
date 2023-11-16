import time
import json
from app import utils
from app.config import Config
from .baseThread import BaseThread
logger = utils.get_logger()

class WebAnalyze(BaseThread):

    def __init__(self, sites, concurrency=3):
        if False:
            print('Hello World!')
        super().__init__(sites, concurrency=concurrency)
        self.analyze_map = {}

    def work(self, site):
        if False:
            i = 10
            return i + 15
        cmd_parameters = ['phantomjs', '--ignore-ssl-errors true', '--ssl-protocol any', '--ssl-ciphers ALL', Config.DRIVER_JS, site]
        logger.debug('WebAnalyze=> {}'.format(' '.join(cmd_parameters)))
        output = utils.check_output(cmd_parameters, timeout=20)
        output = output.decode('utf-8')
        self.analyze_map[site] = json.loads(output)['applications']

    def run(self):
        if False:
            print('Hello World!')
        t1 = time.time()
        logger.info('start WebAnalyze {}'.format(len(self.targets)))
        self._run()
        elapse = time.time() - t1
        logger.info('end WebAnalyze elapse {}'.format(elapse))
        return self.analyze_map

def web_analyze(sites, concurrency=3):
    if False:
        for i in range(10):
            print('nop')
    s = WebAnalyze(sites, concurrency=concurrency)
    return s.run()