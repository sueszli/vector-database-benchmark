"""Classes related to auto-updating and getting the latest version."""
import json
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, QObject, QUrl
from qutebrowser.misc import httpclient

class PyPIVersionClient(QObject):
    """A client for the PyPI API using HTTPClient.

    It gets the latest version of qutebrowser from PyPI.

    Attributes:
        _client: The HTTPClient used.

    Class attributes:
        API_URL: The base API URL.

    Signals:
        success: Emitted when getting the version info succeeded.
                 arg: The newest version.
        error: Emitted when getting the version info failed.
               arg: The error message, as string.
    """
    API_URL = 'https://pypi.org/pypi/{}/json'
    success = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None, client=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        if client is None:
            self._client = httpclient.HTTPClient(self)
        else:
            self._client = client
        self._client.error.connect(self.error)
        self._client.success.connect(self.on_client_success)

    def get_version(self, package='qutebrowser'):
        if False:
            print('Hello World!')
        'Get the newest version of a given package.\n\n        Emits success/error when done.\n\n        Args:\n            package: The name of the package to check.\n        '
        url = QUrl(self.API_URL.format(package))
        self._client.get(url)

    @pyqtSlot(str)
    def on_client_success(self, data):
        if False:
            return 10
        'Process the data and finish when the client finished.\n\n        Args:\n            data: A string with the received data.\n        '
        try:
            json_data = json.loads(data)
        except ValueError as e:
            self.error.emit('Invalid JSON received in reply: {}!'.format(e))
            return
        try:
            self.success.emit(json_data['info']['version'])
        except KeyError as e:
            self.error.emit('Malformed data received in reply ({!r} not found)!'.format(e))
            return