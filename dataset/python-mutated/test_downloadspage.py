from unittest.mock import MagicMock, Mock, patch
from PyQt5.QtWidgets import QWidget
from tribler.gui.network.request import REQUEST_ID
from tribler.gui.widgets.downloadspage import DownloadsPage

def downloads_page() -> DownloadsPage:
    if False:
        while True:
            i = 10
    window = MagicMock()
    window.downloads_list.indexOfTopLevelItem = Mock(return_value=-1)
    page = DownloadsPage()
    page.window = Mock(return_value=window)
    page.received_downloads = Mock()
    return page

@patch.object(QWidget, '__init__', Mock())
def test_accept_requests():
    if False:
        for i in range(10):
            print('nop')
    page = downloads_page()
    page.on_received_downloads(result={REQUEST_ID: 1, 'downloads': MagicMock()})
    assert page.received_downloads.emit.called
    page.received_downloads.emit.reset_mock()
    page.on_received_downloads(result={REQUEST_ID: 2, 'downloads': MagicMock()})
    assert page.received_downloads.emit.called
    page.received_downloads.emit.reset_mock()
    page.on_received_downloads(result={REQUEST_ID: 10, 'downloads': MagicMock()})
    assert page.received_downloads.emit.called

@patch.object(QWidget, '__init__', Mock())
def test_ignore_request():
    if False:
        print('Hello World!')
    page = downloads_page()
    page.on_received_downloads(result={REQUEST_ID: 10, 'downloads': MagicMock()})
    assert page.received_downloads.emit.called
    page.received_downloads.emit.reset_mock()
    page.on_received_downloads(result={REQUEST_ID: 10, 'downloads': MagicMock()})
    assert not page.received_downloads.emit.called
    page.received_downloads.emit.reset_mock()
    page.on_received_downloads(result={REQUEST_ID: 9, 'downloads': MagicMock()})
    assert not page.received_downloads.emit.called