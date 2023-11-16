from unittest.mock import MagicMock
import pytest
from cura.PrinterOutput.Models.PrinterConfigurationModel import PrinterConfigurationModel
from cura.PrinterOutput.Models.ExtruderConfigurationModel import ExtruderConfigurationModel
test_validate_data_get_set = [{'attribute': 'extruderConfigurations', 'value': [ExtruderConfigurationModel()]}, {'attribute': 'buildplateConfiguration', 'value': 'BHDHAHHADAD'}, {'attribute': 'printerType', 'value': ':(', 'check_signal': False}]

@pytest.mark.parametrize('data', test_validate_data_get_set)
def test_getAndSet(data):
    if False:
        print('Hello World!')
    model = PrinterConfigurationModel()
    attribute = list(data['attribute'])
    attribute[0] = attribute[0].capitalize()
    attribute = ''.join(attribute)
    model.configurationChanged = MagicMock()
    signal = model.configurationChanged
    getattr(model, 'set' + attribute)(data['value'])
    if data.get('check_signal', True):
        assert signal.emit.call_count == 1
    assert getattr(model, data['attribute']) == data['value']
    getattr(model, 'set' + attribute)(data['value'])
    if data.get('check_signal', True):
        assert signal.emit.call_count == 1