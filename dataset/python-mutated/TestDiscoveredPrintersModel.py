from unittest.mock import MagicMock, PropertyMock
import pytest
from cura.Machines.Models.DiscoveredPrintersModel import DiscoveredPrintersModel, DiscoveredPrinter

@pytest.fixture()
def discovered_printer_model(application) -> DiscoveredPrintersModel:
    if False:
        return 10
    return DiscoveredPrintersModel(application)

@pytest.fixture()
def discovered_printer() -> DiscoveredPrinter:
    if False:
        return 10
    return DiscoveredPrinter('127.0.0.1', 'zomg', 'yay', None, 'bleep', MagicMock())

@pytest.mark.skip
def test_discoveredPrinters(discovered_printer_model):
    if False:
        print('Hello World!')
    mocked_device = MagicMock()
    cluster_size = PropertyMock(return_value=1)
    type(mocked_device).clusterSize = cluster_size
    mocked_callback = MagicMock()
    discovered_printer_model.addDiscoveredPrinter('ip', 'key', 'name', mocked_callback, 'machine_type', mocked_device)
    device = discovered_printer_model.discoveredPrinters[0]
    discovered_printer_model.createMachineFromDiscoveredPrinter(device)
    mocked_callback.assert_called_with('key')
    assert len(discovered_printer_model.discoveredPrinters) == 1
    discovered_printer_model.discoveredPrintersChanged = MagicMock()
    discovered_printer_model.removeDiscoveredPrinter('ip')
    assert len(discovered_printer_model.discoveredPrinters) == 0
    assert discovered_printer_model.discoveredPrintersChanged.emit.call_count == 1
    discovered_printer_model.removeDiscoveredPrinter('ip')
    assert discovered_printer_model.discoveredPrintersChanged.emit.call_count == 1
test_validate_data_get_set = [{'attribute': 'name', 'value': 'zomg'}, {'attribute': 'machineType', 'value': 'BHDHAHHADAD'}]

@pytest.mark.parametrize('data', test_validate_data_get_set)
def test_getAndSet(data, discovered_printer):
    if False:
        return 10
    attribute = list(data['attribute'])
    attribute[0] = attribute[0].capitalize()
    attribute = ''.join(attribute)
    getattr(discovered_printer, 'set' + attribute)(data['value'])
    assert getattr(discovered_printer, data['attribute']) == data['value']

def test_isHostofGroup(discovered_printer):
    if False:
        print('Hello World!')
    discovered_printer.device.clusterSize = 0
    assert not discovered_printer.isHostOfGroup
    discovered_printer.device.clusterSize = 2
    assert discovered_printer.isHostOfGroup