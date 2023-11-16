import functools
from UM.Qt.Duration import Duration
from cura.UI import PrintInformation
from cura.Settings.MachineManager import MachineManager
from unittest.mock import MagicMock, patch
from UM.MimeTypeDatabase import MimeTypeDatabase, MimeType

def getPrintInformation(printer_name) -> PrintInformation:
    if False:
        i = 10
        return i + 15
    mock_application = MagicMock(name='mock_application')
    mocked_preferences = MagicMock(name='mocked_preferences')
    mocked_extruder_stack = MagicMock()
    mocked_extruder_stack.getProperty = MagicMock(return_value=3)
    mocked_material = MagicMock(name='mocked material')
    mocked_material.getMetaDataEntry = MagicMock(return_value='omgzomg')
    mocked_extruder_stack.material = mocked_material
    mock_application.getInstance = MagicMock(return_value=mock_application)
    mocked_preferences.getValue = MagicMock(return_value='{"omgzomg": {"spool_weight": 10, "spool_cost": 9}}')
    global_container_stack = MagicMock()
    global_container_stack.definition.getName = MagicMock(return_value=printer_name)
    mock_application.getGlobalContainerStack = MagicMock(return_value=global_container_stack)
    mock_application.getPreferences = MagicMock(return_value=mocked_preferences)
    multi_build_plate_model = MagicMock()
    multi_build_plate_model.maxBuildPlate = 0
    mock_application.getMultiBuildPlateModel = MagicMock(return_value=multi_build_plate_model)
    original_get_abbreviated_name = MachineManager.getAbbreviatedMachineName
    mock_machine_manager = MagicMock()
    mock_machine_manager.getAbbreviatedMachineName = functools.partial(original_get_abbreviated_name, mock_machine_manager)
    mock_application.getMachineManager = MagicMock(return_value=mock_machine_manager)
    with patch('UM.Application.Application.getInstance', MagicMock(return_value=mock_application)):
        with patch('json.loads', lambda x: {}):
            print_information = PrintInformation.PrintInformation(mock_application)
    return print_information

def setup_module():
    if False:
        for i in range(10):
            print('nop')
    MimeTypeDatabase.addMimeType(MimeType(name='application/vnd.ms-package.3dmanufacturing-3dmodel+xml', comment='3MF', suffixes=['3mf']))
    MimeTypeDatabase.addMimeType(MimeType(name='application/x-cura-gcode-file', comment='Cura G-code File', suffixes=['gcode']))

def test_duration():
    if False:
        for i in range(10):
            print('nop')
    print_information = getPrintInformation('ultimaker')
    feature_print_times = print_information.getFeaturePrintTimes()
    assert int(feature_print_times['Travel']) == int(Duration(None))
    print_information.setToZeroPrintInformation()
    assert int(feature_print_times['Travel']) == 0
    print_information._onPrintDurationMessage(0, {'travel': 20}, [10])
    assert int(print_information.currentPrintTime) == 20
    feature_print_times = print_information.getFeaturePrintTimes()
    assert int(feature_print_times['Travel']) == 20
    print_information.setToZeroPrintInformation()
    assert int(feature_print_times['Travel']) == 0

def test_setProjectName():
    if False:
        for i in range(10):
            print('nop')
    print_information = getPrintInformation('ultimaker')
    project_name = ['HelloWorld', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['Hello.World', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['Hello.World.World', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['.Hello.World', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['Hello_World', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['Hello_World', '.gcode']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] == print_information._job_name
    project_name = ['', '']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert print_information.UNTITLED_JOB_NAME == print_information._job_name
    project_name = ['Hello_World', '.test']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert 'UM_' + project_name[0] != print_information._job_name

def test_setJobName():
    if False:
        i = 10
        return i + 15
    print_information = getPrintInformation('ultimaker')
    print_information._abbr_machine = 'UM'
    print_information.setJobName('UM_HelloWorld', is_user_specified_job_name=False)

def test_defineAbbreviatedMachineName():
    if False:
        print('Hello World!')
    printer_name = 'Test'
    print_information = getPrintInformation(printer_name)
    project_name = ['HelloWorld', '.3mf']
    print_information.setProjectName(project_name[0] + project_name[1])
    assert printer_name[0] + '_' + project_name[0] == print_information._job_name