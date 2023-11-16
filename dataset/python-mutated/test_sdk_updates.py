import pytest
from sentry.sdk_updates import SdkIndexState, SdkSetupState, get_suggested_updates
PYTHON_INDEX_STATE = SdkIndexState(sdk_versions={'sentry.python': '0.9.1'})
DOTNET_INDEX_STATE = SdkIndexState(sdk_versions={'sentry.dotnet': '1.2.0'})

def test_too_old_django():
    if False:
        return 10
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.9.1', integrations=[], modules={'django': '1.3'})
    assert list(get_suggested_updates(setup, PYTHON_INDEX_STATE)) == []

def test_too_old_sdk():
    if False:
        return 10
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.1.0', integrations=[], modules={'django': '1.8'})
    assert list(get_suggested_updates(setup, PYTHON_INDEX_STATE)) == [{'enables': [{'type': 'enableIntegration', 'enables': [], 'integrationName': 'django', 'integrationUrl': 'https://docs.sentry.io/platforms/python/django/'}], 'newSdkVersion': '0.9.1', 'sdkName': 'sentry.python', 'sdkUrl': None, 'type': 'updateSdk'}]

def test_ignore_patch_version():
    if False:
        print('Hello World!')
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.9.0', integrations=[], modules={})
    assert len(list(get_suggested_updates(setup, PYTHON_INDEX_STATE))) == 0

def test_ignore_patch_version_explicit():
    if False:
        print('Hello World!')
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.9.0', integrations=[], modules={})
    assert len(list(get_suggested_updates(setup, PYTHON_INDEX_STATE, ignore_patch_version=True))) == 0

def test_enable_django_integration():
    if False:
        i = 10
        return i + 15
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.9.1', integrations=[], modules={'django': '1.8'})
    assert list(get_suggested_updates(setup, PYTHON_INDEX_STATE)) == [{'type': 'enableIntegration', 'enables': [], 'integrationName': 'django', 'integrationUrl': 'https://docs.sentry.io/platforms/python/django/'}]

def test_update_sdk():
    if False:
        return 10
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.1.0', integrations=[], modules={})
    assert list(get_suggested_updates(setup, PYTHON_INDEX_STATE)) == [{'enables': [], 'newSdkVersion': '0.9.1', 'sdkName': 'sentry.python', 'sdkUrl': None, 'type': 'updateSdk'}]

def test_enable_two_integrations():
    if False:
        i = 10
        return i + 15
    setup = SdkSetupState(sdk_name='sentry.python', sdk_version='0.1.0', integrations=[], modules={'django': '1.8.0', 'flask': '1.0.0'})
    assert list(get_suggested_updates(setup, PYTHON_INDEX_STATE)) == [{'enables': [{'type': 'enableIntegration', 'enables': [], 'integrationName': 'django', 'integrationUrl': 'https://docs.sentry.io/platforms/python/django/'}, {'type': 'enableIntegration', 'enables': [], 'integrationName': 'flask', 'integrationUrl': 'https://docs.sentry.io/platforms/python/flask/'}], 'newSdkVersion': '0.9.1', 'sdkName': 'sentry.python', 'sdkUrl': None, 'type': 'updateSdk'}]

@pytest.fixture(params=['sentry.dotnet.serilog', 'sentry.dotnet.aspnetcore', 'sentry.dotnet.foobar', 'sentry.dotnet'])
def some_dotnet_sdk(request):
    if False:
        return 10
    return request.param

def test_add_aspnetcore_sdk(some_dotnet_sdk):
    if False:
        for i in range(10):
            print('nop')
    setup = SdkSetupState(sdk_name=some_dotnet_sdk, sdk_version='1.2.0', integrations=[], modules={'Sentry.Serilog': '1.2.0', 'Microsoft.AspNetCore.Hosting': '2.2.0', 'Serilog': '2.7.1'})
    suggestions = list(get_suggested_updates(setup, DOTNET_INDEX_STATE))
    if some_dotnet_sdk == 'sentry.dotnet.aspnetcore':
        assert not suggestions
    else:
        assert suggestions == [{'enables': [], 'newSdkName': 'sentry.dotnet.aspnetcore', 'sdkUrl': None, 'type': 'changeSdk'}]

def test_add_serilog_sdk(some_dotnet_sdk):
    if False:
        i = 10
        return i + 15
    setup = SdkSetupState(sdk_name=some_dotnet_sdk, sdk_version='1.2.0', integrations=[], modules={'Sentry.AspNetCore': '1.2.0', 'Microsoft.AspNetCore': '2.2.0', 'Serilog': '2.7.1'})
    suggestions = list(get_suggested_updates(setup, DOTNET_INDEX_STATE))
    if some_dotnet_sdk == 'sentry.dotnet.serilog':
        assert not suggestions
    else:
        assert suggestions == [{'enables': [], 'newSdkName': 'sentry.dotnet.serilog', 'sdkUrl': None, 'type': 'changeSdk'}]

def test_add_no_dotnet_sdk(some_dotnet_sdk):
    if False:
        while True:
            i = 10
    setup = SdkSetupState(sdk_name=some_dotnet_sdk, sdk_version='1.2.0', integrations=[], modules={'Sentry.AspNetCore': '1.2.0', 'Sentry.Serilog': '1.2.0', 'Microsoft.AspNetCore': '2.2.0', 'Serilog': '2.7.1'})
    suggestions = list(get_suggested_updates(setup, DOTNET_INDEX_STATE))
    assert suggestions == []

def test_more_specific_dotnet_sdk(some_dotnet_sdk):
    if False:
        for i in range(10):
            print('nop')
    setup = SdkSetupState(sdk_name=some_dotnet_sdk, sdk_version='1.2.0', integrations=[], modules={'Microsoft.AspNetCore.Hosting': '2.1.0', 'Microsoft.Extensions.Logging.Configuration': '2.1.0'})
    suggestions = list(get_suggested_updates(setup, DOTNET_INDEX_STATE))
    if some_dotnet_sdk == 'sentry.dotnet.aspnetcore':
        assert not suggestions
    else:
        assert suggestions == [{'enables': [], 'newSdkName': 'sentry.dotnet.aspnetcore', 'sdkUrl': None, 'type': 'changeSdk'}]