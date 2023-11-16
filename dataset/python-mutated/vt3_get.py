from api_app.analyzers_manager.classes import ObservableAnalyzer
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
from .vt3_base import VirusTotalv3AnalyzerMixin

class VirusTotalv3(ObservableAnalyzer, VirusTotalv3AnalyzerMixin):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        result = self._vt_get_report(self.observable_classification, self.observable_name)
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.get', side_effect=[MockUpResponse({'data': {'attributes': {'status': 'completed', 'last_analysis_results': {'test': 'test'}, 'last_analysis_date': 1590000000}}}, 200), MockUpResponse({'data': {'attributes': {'status': 'completed'}}}, 200), MockUpResponse({}, 200), MockUpResponse({}, 200)]), patch('requests.post', return_value=MockUpResponse({'scan_id': 'scan_id_test', 'data': {'id': 'id_test'}}, 200)))]
        return super()._monkeypatch(patches=patches)