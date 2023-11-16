import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerConfigurationException, AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class WiGLE(classes.ObservableAnalyzer):
    base_url: str = 'https://api.wigle.net'
    _api_key_name: str
    search_type: str

    def __prepare_args(self):
        if False:
            return 10
        args = self.observable_name.split(';')
        self.args = {}
        for arg in args:
            try:
                (key, value) = arg.split('=')
            except ValueError:
                key = 'wifiNetworkId'
                value = arg
            self.args[key] = value

    def run(self):
        if False:
            return 10
        self.__prepare_args()
        try:
            if self.search_type == 'WiFi Network':
                uri = f"/api/v3/detail/wifi/{self.args.get('wifiNetworkId')}"
            elif self.search_type == 'CDMA Network':
                uri = f"/api/v3/detail/cell/CDMA/{self.args.get('sid')}/{self.args.get('nid')}/{self.args.get('bsid')}"
            elif self.search_type == 'Bluetooth Network':
                uri = f"/api/v3/detail/bt/{self.args.get('btNetworkId')}"
            elif self.search_type == 'GSM/LTE/WCDMA Network':
                uri = f"/api/v3/detail/cell/{self.args.get('type')}/{self.args.get('operator')}/{self.args.get('lac')}/{self.args.get('cid')}"
            else:
                raise AnalyzerConfigurationException(f"search type: '{self.search_type}' not supported.Supported are: 'WiFi Network', 'CDMA Network', 'Bluetooth Network', 'GSM/LTE/WCDMA Network'")
            response = requests.get(self.base_url + uri, headers={'Authorization': 'Basic ' + self._api_key_name})
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)