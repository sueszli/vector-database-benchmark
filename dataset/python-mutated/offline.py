"""Processors for engine-type: ``offline``

"""
from .abstract import EngineProcessor

class OfflineProcessor(EngineProcessor):
    """Processor class used by ``offline`` engines"""
    engine_type = 'offline'

    def _search_basic(self, query, params):
        if False:
            for i in range(10):
                print('nop')
        return self.engine.search(query, params)

    def search(self, query, params, result_container, start_time, timeout_limit):
        if False:
            for i in range(10):
                print('nop')
        try:
            search_results = self._search_basic(query, params)
            self.extend_container(result_container, start_time, search_results)
        except ValueError as e:
            self.logger.exception('engine {0} : invalid input : {1}'.format(self.engine_name, e))
        except Exception as e:
            self.handle_exception(result_container, e)
            self.logger.exception('engine {0} : exception : {1}'.format(self.engine_name, e))