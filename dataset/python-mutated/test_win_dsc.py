import os.path
import shutil
import subprocess
import psutil
import pytest
import salt.exceptions
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.slow_test]

@pytest.fixture(scope='module')
def dsc(modules):
    if False:
        i = 10
        return i + 15
    existing_config_mode = modules.dsc.get_lcm_config()['ConfigurationMode']
    modules.dsc.set_lcm_config(config_mode='ApplyOnly')
    yield modules.dsc
    modules.dsc.set_lcm_config(config_mode=existing_config_mode)

@pytest.fixture(scope='function')
def ps1_file():
    if False:
        for i in range(10):
            print('nop')
    '\n    This will create a DSC file to be configured. When configured it will create\n    a localhost.mof file in the `HelloWorld` directory in Temp\n    '
    ps1_contents = '\n    Configuration HelloWorld {\n\n        # Import the module that contains the File resource.\n        Import-DscResource -ModuleName PsDesiredStateConfiguration\n\n        # The Node statement specifies which targets to compile MOF files for, when this configuration is executed.\n        Node ("localhost") {\n\n            # The File resource can ensure the state of files, or copy them from a source to a destination with persistent updates.\n            File HelloWorld {\n                DestinationPath = "C:\\Temp\\HelloWorld.txt"\n                Ensure          = "Present"\n                Contents        = "Hello World, ps1_file"\n            }\n        }\n    }\n    '
    with pytest.helpers.temp_file('hello_world.ps1', contents=ps1_contents) as file:
        yield file
    if os.path.exists(file.parent / 'HelloWorld'):
        shutil.rmtree(file.parent / 'HelloWorld')
    if os.path.exists(file):
        os.remove(file)

@pytest.fixture(scope='function')
def ps1_file_multiple():
    if False:
        return 10
    '\n    This will create a DSC file to be configured. When configured it will create\n    a localhost.mof file in the `HelloWorld2` directory in Temp\n    '
    ps1_contents = '\n    Configuration HelloWorldMultiple {\n\n        # Import the module that contains the File resource.\n        Import-DscResource -ModuleName PsDesiredStateConfiguration\n\n        # The Node statement specifies which targets to compile MOF files for, when this configuration is executed.\n        Node ("localhost") {\n\n            # The File resource can ensure the state of files, or copy them from a source to a destination with persistent updates.\n            File HelloWorld {\n                DestinationPath = "C:\\Temp\\HelloWorld.txt"\n                Ensure          = "Present"\n                Contents        = "Hello World from DSC!"\n            }\n\n            # The File resource can ensure the state of files, or copy them from a source to a destination with persistent updates.\n            File HelloWorld2 {\n                DestinationPath = "C:\\Temp\\HelloWorld2.txt"\n                Ensure          = "Present"\n                Contents        = "Hello World, ps1_file_multiple"\n            }\n        }\n    }\n    '
    with pytest.helpers.temp_file('hello_world_multiple.ps1', contents=ps1_contents) as file:
        yield file
    if os.path.exists(file.parent / 'HelloWorldMultiple'):
        shutil.rmtree(file.parent / 'HelloWorldMultiple')
    if os.path.exists(file):
        os.remove(file)

@pytest.fixture(scope='function')
def ps1_file_meta():
    if False:
        for i in range(10):
            print('nop')
    '\n    This will create a DSC file to be configured. When configured it will create\n    a localhost.mof file and a localhost.meta.mof file in the `HelloWorld`\n    directory in Temp\n    '
    ps1_contents = '\n    Configuration HelloWorld {\n\n        # Import the module that contains the File resource.\n        Import-DscResource -ModuleName PsDesiredStateConfiguration\n\n        # The Node statement specifies which targets to compile MOF files for, when this configuration is executed.\n        Node ("localhost") {\n\n            # The File resource can ensure the state of files, or copy them from a source to a destination with persistent updates.\n            File HelloWorld {\n                DestinationPath = "C:\\Temp\\HelloWorld.txt"\n                Ensure          = "Present"\n                Contents        = "Hello World, ps1_file_meta "\n            }\n\n            # Set some Meta Config\n            LocalConfigurationManager {\n                ConfigurationMode  = "ApplyAndMonitor"\n                RebootNodeIfNeeded = $false\n                RefreshMode        = "PUSH"\n            }\n        }\n    }\n    '
    with pytest.helpers.temp_file('test.ps1', contents=ps1_contents) as file:
        yield file
    if os.path.exists(file.parent / 'HelloWorld'):
        shutil.rmtree(file.parent / 'HelloWorld')
    if os.path.exists(file):
        os.remove(file)

@pytest.fixture(scope='module')
def psd1_file():
    if False:
        i = 10
        return i + 15
    '\n    This will create a config data file to be applied with the config file in\n    Temp\n    '
    psd1_contents = "\n    @{\n        AllNodes = @(\n            @{\n                NodeName = 'localhost'\n                PSDscAllowPlainTextPassword = $true\n                PSDscAllowDomainUser = $true\n            }\n        )\n    }\n    "
    with pytest.helpers.temp_file('test.psd1', contents=psd1_contents) as file:
        yield file
    if os.path.exists(file):
        os.remove(file)

def test_compile_config_missing(dsc):
    if False:
        i = 10
        return i + 15
    path = 'C:\\Path\\not\\exists.ps1'
    with pytest.raises(salt.exceptions.CommandExecutionError) as exc:
        dsc.compile_config(path=path)
    assert exc.value.message == '{} not found'.format(path)

@pytest.mark.destructive_test
def test_compile_config(dsc, ps1_file, psd1_file):
    if False:
        print('Hello World!')
    '\n    Test compiling a simple config\n    '
    dsc.remove_config(reset=False)
    result = dsc.compile_config(path=str(ps1_file), config_name='HelloWorld', config_data=str(psd1_file))
    assert isinstance(result, dict)
    assert result['Exists'] is True

@pytest.mark.destructive_test
def test_compile_config_issue_61261(dsc, ps1_file_meta, psd1_file):
    if False:
        return 10
    '\n    Test compiling a config that includes meta data\n    '
    dsc.remove_config(reset=False)
    result = dsc.compile_config(path=str(ps1_file_meta), config_name='HelloWorld', config_data=str(psd1_file))
    assert isinstance(result, dict)
    assert result['Exists'] is True

def test_apply_config_missing(dsc):
    if False:
        print('Hello World!')
    path = 'C:\\Path\\not\\exists'
    with pytest.raises(salt.exceptions.CommandExecutionError) as exc:
        dsc.apply_config(path=path)
    assert exc.value.message == '{} not found'.format(path)

@pytest.mark.destructive_test
def test_apply_config(dsc, ps1_file, psd1_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test applying a simple config\n    '
    dsc.remove_config(reset=False)
    dsc.compile_config(path=str(ps1_file), config_name='HelloWorld', config_data=str(psd1_file))
    result = dsc.apply_config(path=ps1_file.parent / 'HelloWorld')
    assert result is True

def test_get_config_not_configured(dsc):
    if False:
        i = 10
        return i + 15
    dsc.remove_config(reset=False)
    with pytest.raises(salt.exceptions.CommandExecutionError) as exc:
        dsc.get_config()
    assert exc.value.message == 'Not Configured'

def test_get_config_single(dsc, ps1_file, psd1_file):
    if False:
        for i in range(10):
            print('nop')
    dsc.remove_config(reset=False)
    dsc.run_config(path=str(ps1_file), config_name='HelloWorld', config_data=str(psd1_file))
    result = dsc.get_config()
    assert 'HelloWorld' in result
    assert '[File]HelloWorld' in result['HelloWorld']
    assert 'DestinationPath' in result['HelloWorld']['[File]HelloWorld']

def test_get_config_multiple(dsc, ps1_file_multiple, psd1_file):
    if False:
        for i in range(10):
            print('nop')
    dsc.remove_config(reset=False)
    dsc.run_config(path=str(ps1_file_multiple), config_name='HelloWorldMultiple', config_data=str(psd1_file))
    result = dsc.get_config()
    assert 'HelloWorldMultiple' in result
    assert '[File]HelloWorld' in result['HelloWorldMultiple']
    assert 'DestinationPath' in result['HelloWorldMultiple']['[File]HelloWorld']
    assert '[File]HelloWorld2' in result['HelloWorldMultiple']
    assert 'DestinationPath' in result['HelloWorldMultiple']['[File]HelloWorld2']

def _reset_config(dsc):
    if False:
        return 10
    '\n    Resets the DSC config. If files are locked, this will attempt to kill the\n    all running WmiPrvSE processes. Windows will respawn the ones it needs\n    '
    tries = 1
    while True:
        try:
            tries += 1
            dsc.remove_config(reset=True)
            break
        except salt.exceptions.CommandExecutionError:
            if tries > 12:
                raise
            proc_name = 'wmiprvse.exe'
            for proc in psutil.process_iter():
                if proc.name().lower() == proc_name:
                    proc.kill()
            continue

def test_get_config_status_not_configured(dsc):
    if False:
        for i in range(10):
            print('nop')
    _reset_config(dsc)
    with pytest.raises(salt.exceptions.CommandExecutionError) as exc:
        dsc.get_config_status()
    assert exc.value.message == 'Not Configured'

def test_get_config_status(dsc, ps1_file, psd1_file):
    if False:
        while True:
            i = 10
    dsc.remove_config(reset=False)
    dsc.run_config(path=str(ps1_file), config_name='HelloWorld', config_data=str(psd1_file))
    result = dsc.get_config_status()
    assert 'MetaData' in result
    assert 'HelloWorld' in result['MetaData']
    assert result['Status'] == 'Success'

def test_test_config_not_configured(dsc):
    if False:
        print('Hello World!')
    subprocess.run(['cmd', '/c', 'winrm', 'quickconfig', '-quiet'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dsc.remove_config(reset=False)
    with pytest.raises(salt.exceptions.CommandExecutionError) as exc:
        dsc.test_config()
    assert exc.value.message == 'Not Configured'

def test_test_config(dsc, ps1_file, psd1_file):
    if False:
        while True:
            i = 10
    dsc.remove_config(reset=False)
    dsc.run_config(path=str(ps1_file), config_name='HelloWorld', config_data=str(psd1_file))
    result = dsc.test_config()
    assert result is True

def test_get_lcm_config(dsc):
    if False:
        i = 10
        return i + 15
    config_items = ['ConfigurationModeFrequencyMins', 'LCMState', 'RebootNodeIfNeeded', 'ConfigurationMode', 'ActionAfterReboot', 'RefreshMode', 'CertificateID', 'ConfigurationID', 'RefreshFrequencyMins', 'AllowModuleOverwrite', 'DebugMode', 'StatusRetentionTimeInDays']
    dsc.remove_config(reset=False)
    result = dsc.get_lcm_config()
    for item in config_items:
        assert item in result

def test_set_lcm_config(dsc):
    if False:
        i = 10
        return i + 15
    current = dsc.get_lcm_config()['ConfigurationMode']
    dsc.set_lcm_config(config_mode='ApplyOnly')
    try:
        results = dsc.get_lcm_config()
        assert results['ConfigurationMode'] == 'ApplyOnly'
    finally:
        dsc.set_lcm_config(config_mode=current)