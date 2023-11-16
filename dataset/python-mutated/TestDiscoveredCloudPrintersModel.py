from unittest.mock import MagicMock
import pytest
from cura.Machines.Models.DiscoveredCloudPrintersModel import DiscoveredCloudPrintersModel

@pytest.fixture()
def discovered_cloud_printers_model(application) -> DiscoveredCloudPrintersModel:
    if False:
        while True:
            i = 10
    return DiscoveredCloudPrintersModel(application)

def test_discoveredCloudPrinters(discovered_cloud_printers_model):
    if False:
        return 10
    new_devices = [{'key': 'Bite my shiny metal a$$', 'name': 'Bender', 'machine_type': 'Bender robot', 'firmware_version': '8.0.0.8.5'}]
    discovered_cloud_printers_model.cloudPrintersDetectedChanged = MagicMock()
    discovered_cloud_printers_model.addDiscoveredCloudPrinters(new_devices)
    assert len(discovered_cloud_printers_model._discovered_cloud_printers_list) == 1
    assert discovered_cloud_printers_model.cloudPrintersDetectedChanged.emit.call_count == 1
    discovered_cloud_printers_model.cloudPrintersDetectedChanged.emit.assert_called_with(True)
    discovered_cloud_printers_model.clear()
    assert len(discovered_cloud_printers_model._discovered_cloud_printers_list) == 0
    assert discovered_cloud_printers_model.cloudPrintersDetectedChanged.emit.call_count == 2
    discovered_cloud_printers_model.cloudPrintersDetectedChanged.emit.assert_called_with(False)