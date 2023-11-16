from pyhashlookup import Hashlookup, PyHashlookupError
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockResponseNoOp, if_mock_connections, patch

class HashLookupServer(classes.ObservableAnalyzer):
    hashlookup_server: str

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.hashlookup_server:
            hashlookup_instance = Hashlookup(root_url=self.hashlookup_server)
        else:
            hashlookup_instance = Hashlookup()
        try:
            result = hashlookup_instance.lookup(self.observable_name)
        except PyHashlookupError as e:
            raise AnalyzerRunException(e)
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('pyhashlookup.Hashlookup', return_value=MockResponseNoOp({}, 200)))]
        return super()._monkeypatch(patches=patches)