from unittest.mock import MagicMock
import pytest
from cura.PrinterOutput.Models.PrinterConfigurationModel import PrinterConfigurationModel
from cura.PrinterOutput.Models.PrintJobOutputModel import PrintJobOutputModel
from cura.PrinterOutput.Models.PrinterOutputModel import PrinterOutputModel
test_validate_data_get_set = [{'attribute': 'compatibleMachineFamilies', 'value': ['yay']}]
test_validate_data_get_update = [{'attribute': 'configuration', 'value': PrinterConfigurationModel()}, {'attribute': 'owner', 'value': 'WHOO'}, {'attribute': 'assignedPrinter', 'value': PrinterOutputModel(MagicMock())}, {'attribute': 'key', 'value': 'YAY'}, {'attribute': 'name', 'value': 'Turtles'}, {'attribute': 'timeTotal', 'value': 10}, {'attribute': 'timeElapsed', 'value': 20}, {'attribute': 'state', 'value': 'BANANNA!'}]

@pytest.mark.parametrize('data', test_validate_data_get_set)
def test_getAndSet(data):
    if False:
        return 10
    model = PrintJobOutputModel(MagicMock())
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
    model = PrintJobOutputModel(MagicMock())
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