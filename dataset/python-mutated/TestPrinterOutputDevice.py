import pytest
from unittest.mock import MagicMock, patch
from cura.PrinterOutput.Models.ExtruderConfigurationModel import ExtruderConfigurationModel
from cura.PrinterOutput.Models.MaterialOutputModel import MaterialOutputModel
from cura.PrinterOutput.Models.PrinterConfigurationModel import PrinterConfigurationModel
from cura.PrinterOutput.Models.PrinterOutputModel import PrinterOutputModel
from cura.PrinterOutput.PrinterOutputDevice import PrinterOutputDevice
test_validate_data_get_set = [{'attribute': 'connectionText', 'value': 'yay'}, {'attribute': 'connectionState', 'value': 1}]

@pytest.fixture()
def printer_output_device():
    if False:
        for i in range(10):
            print('nop')
    with patch('UM.Application.Application.getInstance'):
        return PrinterOutputDevice('whatever')

@pytest.mark.parametrize('data', test_validate_data_get_set)
def test_getAndSet(data, printer_output_device):
    if False:
        while True:
            i = 10
    model = printer_output_device
    attribute = list(data['attribute'])
    attribute[0] = attribute[0].capitalize()
    attribute = ''.join(attribute)
    setattr(model, data['attribute'] + 'Changed', MagicMock())
    with patch('cura.CuraApplication.CuraApplication.getInstance'):
        getattr(model, 'set' + attribute)(data['value'])
    signal = getattr(model, data['attribute'] + 'Changed')
    assert signal.emit.call_count == 1
    assert getattr(model, data['attribute']) == data['value']
    getattr(model, 'set' + attribute)(data['value'])
    assert signal.emit.call_count == 1

def test_uniqueConfigurations(printer_output_device):
    if False:
        while True:
            i = 10
    printer = PrinterOutputModel(MagicMock())
    printer_output_device._printers = [printer]
    printer_output_device._onPrintersChanged()
    assert printer_output_device.uniqueConfigurations == []
    configuration = PrinterConfigurationModel()
    printer.addAvailableConfiguration(configuration)
    assert printer_output_device.uniqueConfigurations == [configuration]
    printer.updateType('blarg!')
    loaded_material = MaterialOutputModel(guid='', type='PLA', color='Blue', brand='Generic', name='Blue PLA')
    loaded_left_extruder = ExtruderConfigurationModel(0)
    loaded_left_extruder.setMaterial(loaded_material)
    loaded_right_extruder = ExtruderConfigurationModel(1)
    loaded_right_extruder.setMaterial(loaded_material)
    printer.printerConfiguration.setExtruderConfigurations([loaded_left_extruder, loaded_right_extruder])
    assert set(printer_output_device.uniqueConfigurations) == set([configuration, printer.printerConfiguration])

def test_uniqueConfigurations_empty_is_filtered_out(printer_output_device):
    if False:
        print('Hello World!')
    printer = PrinterOutputModel(MagicMock())
    printer_output_device._printers = [printer]
    printer_output_device._onPrintersChanged()
    printer.updateType('blarg!')
    empty_material = MaterialOutputModel(guid='', type='empty', color='empty', brand='Generic', name='Empty')
    empty_left_extruder = ExtruderConfigurationModel(0)
    empty_left_extruder.setMaterial(empty_material)
    empty_right_extruder = ExtruderConfigurationModel(1)
    empty_right_extruder.setMaterial(empty_material)
    printer.printerConfiguration.setExtruderConfigurations([empty_left_extruder, empty_right_extruder])
    assert printer_output_device.uniqueConfigurations == []