import logging
import socket
from api_app.analyzers_manager.classes import ObservableAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException
logger = logging.getLogger(__name__)

class Cymru(ObservableAnalyzer):

    def run(self):
        if False:
            print('Hello World!')
        results = {}
        if self.observable_classification != self.ObservableTypes.HASH:
            raise AnalyzerRunException(f'observable type {self.observable_classification} not supported')
        hash_length = len(self.observable_name)
        if hash_length == 64:
            raise AnalyzerRunException('sha256 are not supported by the service')
        results['found'] = False
        domains = None
        try:
            query_to_perform = f'{self.observable_name}.malware.hash.cymru.com'
            domains = socket.gethostbyaddr(query_to_perform)
        except (socket.gaierror, socket.herror):
            logger.info(f'observable {self.observable_name} not found in HMR DB')
        if domains:
            results['found'] = True
            results['resolution_data'] = domains[2]
        return results