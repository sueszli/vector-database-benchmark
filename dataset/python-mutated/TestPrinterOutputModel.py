from unittest.mock import MagicMock
import pytest
from cura.PrinterOutput.Models.PrintJobOutputModel import PrintJobOutputModel
from cura.PrinterOutput.Models.PrinterConfigurationModel import PrinterConfigurationModel
from cura.PrinterOutput.Models.PrinterOutputModel import PrinterOutputModel
from cura.PrinterOutput.Peripheral import Peripheral
test_validate_data_get_set = [{'attribute': 'name', 'value': 'YAY'}, {'attribute': 'targetBedTemperature', 'value': 192}, {'attribute': 'cameraUrl', 'value': 'YAY!'}]
test_validate_data_get_update = [{'attribute': 'isPreheating', 'value': True}, {'attribute': 'type', 'value': 'WHOO'}, {'attribute': 'buildplate', 'value': 'NFHA'}, {'attribute': 'key', 'value': 'YAY'}, {'attribute': 'name', 'value': 'Turtles'}, {'attribute': 'bedTemperature', 'value': 200}, {'attribute': 'targetBedTemperature', 'value': 9001}, {'attribute': 'activePrintJob', 'value': PrintJobOutputModel(MagicMock())}, {'attribute': 'state', 'value': 'BEEPBOOP'}]

@pytest.mark.parametrize('data', test_validate_data_get_set)
def test_getAndSet(data):
    if False:
        print('Hello World!')
    model = PrinterOutputModel(MagicMock())
    attribute = list(data['attribute'])
    attribute[0] = attribute[0].capitalize()
    attribute = ''.join(attribute)
    setattr(model, data['attribute'] + 'Changed', MagicMock())
    getattr(model, 'set' + attribute)(data['value'])
    signal = getattr(model, data['attribute'] + 'Changed')
    assert signal.emit.call_count == 1
    assert getattr(model, data['attribute']) == data['value']
    getattr(model, 'set' + attribute)(data['value'])
    assert signal.emit.call_count == 1

@pytest.mark.parametrize('data', test_validate_data_get_update)
def test_getAndUpdate(data):
    if False:
        while True:
            i = 10
    model = PrinterOutputModel(MagicMock())
    attribute = list(data['attribute'])
    attribute[0] = attribute[0].capitalize()
    attribute = ''.join(attribute)
    setattr(model, data['attribute'] + 'Changed', MagicMock())
    getattr(model, 'update' + attribute)(data['value'])
    signal = getattr(model, data['attribute'] + 'Changed')
    assert signal.emit.call_count == 1
    assert getattr(model, data['attribute']) == data['value']
    getattr(model, 'update' + attribute)(data['value'])
    assert signal.emit.call_count == 1

def test_peripherals():
    if False:
        return 10
    model = PrinterOutputModel(MagicMock())
    model.peripheralsChanged = MagicMock()
    peripheral = MagicMock(spec=Peripheral)
    peripheral.name = 'test'
    peripheral2 = MagicMock(spec=Peripheral)
    peripheral2.name = 'test2'
    model.addPeripheral(peripheral)
    assert model.peripheralsChanged.emit.call_count == 1
    model.addPeripheral(peripheral2)
    assert model.peripheralsChanged.emit.call_count == 2
    assert model.peripherals == 'test, test2'
    model.removePeripheral(peripheral)
    assert model.peripheralsChanged.emit.call_count == 3
    assert model.peripherals == 'test2'

def test_availableConfigurations_addConfiguration():
    if False:
        while True:
            i = 10
    model = PrinterOutputModel(MagicMock())
    configuration = MagicMock(spec=PrinterConfigurationModel)
    model.addAvailableConfiguration(configuration)
    assert model.availableConfigurations == [configuration]

def test_availableConfigurations_addConfigTwice():
    if False:
        while True:
            i = 10
    model = PrinterOutputModel(MagicMock())
    configuration = MagicMock(spec=PrinterConfigurationModel)
    model.setAvailableConfigurations([configuration])
    assert model.availableConfigurations == [configuration]
    model.addAvailableConfiguration(configuration)
    assert model.availableConfigurations == [configuration]

def test_availableConfigurations_removeConfig():
    if False:
        return 10
    model = PrinterOutputModel(MagicMock())
    configuration = MagicMock(spec=PrinterConfigurationModel)
    model.addAvailableConfiguration(configuration)
    model.removeAvailableConfiguration(configuration)
    assert model.availableConfigurations == []

def test_removeAlreadyRemovedConfiguration():
    if False:
        return 10
    model = PrinterOutputModel(MagicMock())
    configuration = MagicMock(spec=PrinterConfigurationModel)
    model.availableConfigurationsChanged = MagicMock()
    model.removeAvailableConfiguration(configuration)
    assert model.availableConfigurationsChanged.emit.call_count == 0
    assert model.availableConfigurations == []