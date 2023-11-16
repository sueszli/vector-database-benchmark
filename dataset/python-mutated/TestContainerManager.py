from unittest import TestCase
from unittest.mock import MagicMock
from PyQt6.QtCore import QUrl
from unittest.mock import patch
from UM.MimeTypeDatabase import MimeTypeDatabase
from cura.Settings.ContainerManager import ContainerManager
import tempfile
import os

class TestContainerManager(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._application = MagicMock()
        self._container_registry = MagicMock()
        self._machine_manager = MagicMock()
        self._machine_manager.activeMachine.extruderList = [MagicMock(name='Left Extruder Mock'), MagicMock(name='Right Extruder Mock')]
        self._mocked_mime = MagicMock()
        self._mocked_mime.preferredSuffix = 'omg'
        self._mocked_mime.suffixes = ['omg']
        self._mocked_mime.comment = 'UnitTest!'
        self._mocked_container = MagicMock()
        self._mocked_container_data = 'SOME DATA :D'
        self._mocked_container.serialize = MagicMock(return_value=self._mocked_container_data)
        self._containers_meta_data = [{'id': 'test', 'test_data': 'omg'}]
        self._container_registry.findContainersMetadata = MagicMock(return_value=self._containers_meta_data)
        self._container_registry.getMimeTypeForContainer = MagicMock(return_value=self._mocked_mime)
        self._container_registry.findContainers = MagicMock(return_value=[self._mocked_container])
        self._application.getContainerRegistry = MagicMock(return_value=self._container_registry)
        self._application.getMachineManager = MagicMock(return_value=self._machine_manager)
        if ContainerManager.getInstance() is not None:
            ContainerManager._ContainerManager__instance = None
        self._container_manager = ContainerManager(self._application)
        MimeTypeDatabase.addMimeType(self._mocked_mime)

    def tearDown(self):
        if False:
            return 10
        MimeTypeDatabase.removeMimeType(self._mocked_mime)

    def test_getContainerMetaDataEntry(self):
        if False:
            print('Hello World!')
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=self._application)):
            assert self._container_manager.getContainerMetaDataEntry('test', 'test_data') == 'omg'
            assert self._container_manager.getContainerMetaDataEntry('test', 'entry_that_is_not_defined') == ''

    def test_clearUserContainer(self):
        if False:
            return 10
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=self._application)):
            self._container_manager.clearUserContainers()
        assert self._machine_manager.activeMachine.userChanges.clear.call_count == 1

    def test_getContainerNameFilters(self):
        if False:
            i = 10
            return i + 15
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=self._application)):
            assert self._container_manager.getContainerNameFilters('') == ['All Files (*)']
            self._container_registry.getContainerTypes = MagicMock(return_value=[('None', None)])
            assert self._container_manager.getContainerNameFilters('') == ['UnitTest! (*.omg)', 'All Files (*)']

    def test_exportContainerUnknownFileType(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._container_manager.exportContainer('test', 'zomg', 'whatever')['status'] == 'error'

    def test_exportContainerInvalidPath(self):
        if False:
            while True:
                i = 10
        assert self._container_manager.exportContainer('test', 'zomg', '')['status'] == 'error'
        assert self._container_manager.exportContainer('test', 'zomg', QUrl())['status'] == 'error'

    def test_exportContainerInvalidId(self):
        if False:
            return 10
        assert self._container_manager.exportContainer('', 'whatever', 'whatever')['status'] == 'error'

    def test_exportContainer(self):
        if False:
            while True:
                i = 10
        with patch('cura.CuraApplication.CuraApplication.getInstance', MagicMock(return_value=self._application)):
            with tempfile.TemporaryDirectory() as tmpdirname:
                result = self._container_manager.exportContainer('test', 'whatever', os.path.join(tmpdirname, 'whatever.omg'))
                assert os.path.exists(result['path'])
                with open(result['path'], 'r', encoding='utf-8') as f:
                    assert f.read() == self._mocked_container_data