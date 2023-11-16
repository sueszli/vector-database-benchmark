"""
:codeauthor: Shane Lee <slee@saltstack.com>
"""
import glob
import logging
import os
import pathlib
import re
import shutil
import zipfile
import pytest
import requests
import salt.grains.core
import salt.modules.win_file as win_file
import salt.modules.win_lgpo as win_lgpo
import salt.utils.files
import salt.utils.win_dacl as win_dacl
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.slow_test]

@pytest.fixture
def configure_loader_modules(tmp_path):
    if False:
        while True:
            i = 10
    cachedir = tmp_path / '__test_admx_policy_cache_dir'
    cachedir.mkdir(parents=True, exist_ok=True)
    return {win_lgpo: {'__salt__': {'file.file_exists': win_file.file_exists, 'file.makedirs': win_file.makedirs_}, '__opts__': {'cachedir': str(cachedir)}}, win_file: {'__utils__': {'dacl.set_perms': win_dacl.set_perms}}}

@pytest.fixture(scope='module')
def osrelease():
    if False:
        for i in range(10):
            print('nop')
    grains = salt.grains.core.os_data()
    yield grains.get('osrelease', None)

@pytest.fixture
def clean_comp():
    if False:
        return 10
    reg_pol = pathlib.Path(os.getenv('SystemRoot'), 'System32', 'GroupPolicy', 'Machine', 'Registry.pol')
    reg_pol.unlink(missing_ok=True)
    yield reg_pol
    reg_pol.unlink(missing_ok=True)

@pytest.fixture
def clean_user():
    if False:
        print('Hello World!')
    reg_pol = pathlib.Path(os.getenv('SystemRoot'), 'System32', 'GroupPolicy', 'User', 'Registry.pol')
    reg_pol.unlink(missing_ok=True)
    yield reg_pol
    reg_pol.unlink(missing_ok=True)

@pytest.fixture(scope='module')
def lgpo_bin():
    if False:
        for i in range(10):
            print('nop')
    sys_dir = pathlib.Path(os.getenv('SystemRoot'), 'System32')
    if not (sys_dir / 'lgpo.exe').exists():
        zip_file = sys_dir / 'lgpo.zip'
        log.debug('Downloading LGPO.exe from Microsoft')
        url = 'https://download.microsoft.com/download/8/5/C/85C25433-A1B0-4FFA-9429-7E023E7DA8D8/LGPO.zip'
        r = requests.get(url)
        with salt.utils.files.fopen(zip_file, 'wb') as f:
            f.write(r.content)
        log.debug('Extracting LGPO.exe')
        with zipfile.ZipFile(zip_file) as z:
            for file in z.namelist():
                if file.lower().endswith('lgpo.exe'):
                    location = z.extract(file, path='C:\\Windows\\System32')
        log.debug('Placing LGPO.exe in System32')
        lgpo_bin = pathlib.Path(location)
        lgpo_bin = lgpo_bin.rename(sys_dir / lgpo_bin.name.lower())
        yield lgpo_bin
        log.debug('Cleaning up LGPO artifacts')
        zip_file.unlink(missing_ok=True)
        lgpo_bin.unlink(missing_ok=True)
        if (sys_dir / 'LGPO_30').exists():
            shutil.rmtree(str(sys_dir / 'LGPO_30'))
    else:
        log.debug('LGPO.exe already present')
        yield str(sys_dir / 'lgpo.exe')

@pytest.mark.destructive_test
def test__load_policy_definitions():
    if False:
        print('Hello World!')
    "\n    Test that unexpected files in the PolicyDefinitions directory won't\n    cause the _load_policy_definitions function to explode\n    https://gitlab.com/saltstack/enterprise/lock/issues/3826\n    "
    bogus_fle = os.path.join('c:\\Windows\\PolicyDefinitions', '_bogus.adml')
    cache_dir = os.path.join(win_lgpo.__opts__['cachedir'], 'lgpo', 'policy_defs')
    try:
        with salt.utils.files.fopen(bogus_fle, 'w+') as fh:
            fh.write('<junk></junk>')
        win_lgpo._load_policy_definitions()
        assert True
    finally:
        os.remove(bogus_fle)
        search_string = f'{cache_dir}\\_bogus*.adml'
        for file_name in glob.glob(search_string):
            os.remove(file_name)

def _test_set_computer_policy(lgpo_bin, shell, name, setting, exp_regexes):
    if False:
        i = 10
        return i + 15
    result = win_lgpo.set_computer_policy(name=name, setting=setting)
    assert result is True
    ret = shell.run(lgpo_bin, '/parse', '/m', 'C:\\Windows\\System32\\GroupPolicy\\Machine\\Registry.pol')
    assert ret.returncode == 0
    content = ret.stdout
    assert re.search('Invalid file format\\.', content, re.IGNORECASE) is None
    for exp_regex in exp_regexes:
        match = re.search(exp_regex, content, re.IGNORECASE | re.MULTILINE)
        assert match is not None

def _test_set_user_policy(lgpo_bin, shell, name, setting, exp_regexes):
    if False:
        return 10
    result = win_lgpo.set_user_policy(name=name, setting=setting)
    assert result is True
    ret = shell.run(lgpo_bin, '/parse', '/u', 'C:\\Windows\\System32\\GroupPolicy\\User\\Registry.pol')
    assert ret.returncode == 0
    content = ret.stdout
    assert re.search('Invalid file format\\.', content, re.IGNORECASE) is None
    for exp_regex in exp_regexes:
        match = re.search(exp_regex, content, re.IGNORECASE | re.MULTILINE)
        assert match is not None

@pytest.mark.parametrize('name, setting, exp_regexes', [('Configure Windows NTP Client', 'Disabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\Parameters[\\s]*NtpServer[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\Parameters[\\s]*Type[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*CrossSiteSyncFlags[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*ResolvePeerBackoffMinutes[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*ResolvePeerBackoffMaxTimes[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*SpecialPollInterval[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*EventLogFlags[\\s]*DELETE']), ('Configure Windows NTP Client', {'NtpServer': 'time.windows.com,0x9', 'Type': 'NT5DS', 'CrossSiteSyncFlags': 2, 'ResolvePeerBackoffMinutes': 15, 'ResolvePeerBackoffMaxTimes': 7, 'W32TIME_SpecialPollInterval': 3600, 'W32TIME_NtpClientEventLogFlags': 0}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\Parameters[\\s]*NtpServer[\\s]*SZ:time.windows.com,0x9', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\Parameters[\\s]*Type[\\s]*SZ:NT5DS', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*CrossSiteSyncFlags[\\s]*DWORD:2', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*ResolvePeerBackoffMinutes[\\s]*DWORD:15', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*ResolvePeerBackoffMaxTimes[\\s]*DWORD:7', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*SpecialPollInterval[\\s]*DWORD:3600', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\W32time\\\\TimeProviders\\\\NtpClient[\\s]*EventLogFlags[\\s]*DWORD:0']), ('Configure Windows NTP Client', 'Not Configured', ['; Source file:  C:\\\\Windows\\\\System32\\\\GroupPolicy\\\\Machine\\\\Registry.pol[\\s]*; PARSING COMPLETED.']), ('RA_Unsolicit', 'Disabled', ['Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicited[\\s]*DWORD:0', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicitedFullControl[\\s]*DELETE', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*\\*[\\s]*DELETEALLVALUES']), ('RA_Unsolicit', {'Permit remote control of this computer': 'Allow helpers to remotely control the computer', 'Helpers': ['administrators', 'user1']}, ['Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*user1[\\s]*SZ:user1[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*administrators[\\s]*SZ:administrators[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicited[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicitedFullControl[\\s]*DWORD:1']), ('RA_Unsolicit', 'Not Configured', ['; Source file:  C:\\\\Windows\\\\System32\\\\GroupPolicy\\\\Machine\\\\Registry.pol[\\s]*; PARSING COMPLETED.']), ('Pol_HardenedPaths', 'Disabled', ['Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows\\\\NetworkProvider\\\\HardenedPaths[\\s]*\\*[\\s]*DELETEALLVALUES']), ('Pol_HardenedPaths', {'Hardened UNC Paths': {'\\\\*\\NETLOGON': 'RequireMutualAuthentication=1, RequireIntegrity=1', '\\\\*\\SYSVOL': 'RequireMutualAuthentication=1, RequireIntegrity=1'}}, ['Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows\\\\NetworkProvider\\\\HardenedPaths[\\s]*\\\\\\\\\\*\\\\NETLOGON[\\s]*SZ:RequireMutualAuthentication=1, RequireIntegrity=1[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows\\\\NetworkProvider\\\\HardenedPaths[\\s]*\\\\\\\\\\*\\\\SYSVOL[\\s]*SZ:RequireMutualAuthentication=1, RequireIntegrity=1[\\s]*']), ('Pol_HardenedPaths', 'Not Configured', ['; Source file:  C:\\\\Windows\\\\System32\\\\GroupPolicy\\\\Machine\\\\Registry.pol[\\s]*; PARSING COMPLETED.']), ('TS_CLIENT_CLIPBOARD', 'Enabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fDisableClip[\\s]*DWORD:1']), ('TS_CLIENT_CLIPBOARD', 'Disabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fDisableClip[\\s]*DWORD:0']), ('Windows Components\\Remote Desktop Services\\Remote Desktop Session Host\\Device and Resource Redirection\\Do not allow Clipboard redirection', 'Not Configured', ['; Source file:  C:\\\\Windows\\\\System32\\\\GroupPolicy\\\\Machine\\\\Registry.pol[\\s]*; PARSING COMPLETED.']), ('DisableUXWUAccess', 'Enabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*SetDisableUXWUAccess[\\s]*DWORD:1']), ('Remove access to use all Windows Update features', 'Disabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*SetDisableUXWUAccess[\\s]*DWORD:0']), ('ActiveHours', {'ActiveHoursStartTime': '8 AM', 'ActiveHoursEndTime': '7 PM'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*SetActiveHours[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*ActiveHoursStart[\\s]*DWORD:8', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*ActiveHoursEnd[\\s]*DWORD:19']), ('Turn off auto-restart for updates during active hours', 'Disabled', ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*SetActiveHours[\\s]*DWORD:0', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*ActiveHoursStart[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate[\\s]*ActiveHoursEnd[\\s]*DELETE']), ('Specify settings for optional component installation and component repair', 'Disabled', ['Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*LocalSourcePath[\\s]*DELETE', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*UseWindowsUpdate[\\s]*DELETE', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*RepairContentServerSource[\\s]*DELETE']), ('Specify settings for optional component installation and component repair', {'Alternate source file path': '', 'Never attempt to download payload from Windows Update': True, 'CheckBox_SidestepWSUS': False}, ['Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*LocalSourcePath[\\s]*EXSZ:', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*UseWindowsUpdate[\\s]*DWORD:2', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*RepairContentServerSource[\\s]*DELETE']), ('Specify settings for optional component installation and component repair', {'Alternate source file path': '\\\\some\\fake\\server', 'Never attempt to download payload from Windows Update': True, 'CheckBox_SidestepWSUS': False}, ['Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*LocalSourcePath[\\s]*EXSZ:\\\\\\\\\\\\\\\\some\\\\\\\\fake\\\\\\\\server', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*UseWindowsUpdate[\\s]*DWORD:2', 'Computer[\\s]*Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Policies\\\\Servicing[\\s]*RepairContentServerSource[\\s]*DELETE']), ('Specify settings for optional component installation and component repair', 'Not Configured', ['; Source file:  C:\\\\Windows\\\\System32\\\\GroupPolicy\\\\Machine\\\\Registry.pol[\\s]*; PARSING COMPLETED.']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Internet Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\Zones\\\\3[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Locked-Down Internet Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\lockdown_zones\\\\3[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Intranet Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\Zones\\\\1[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Locked-Down Intranet Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\lockdown_zones\\\\1[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Local Machine Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\Zones\\\\0[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Locked-Down Local Machine Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\lockdown_zones\\\\0[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Restricted Sites Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\Zones\\\\4[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Locked-Down Restricted Sites Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\lockdown_zones\\\\4[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Trusted Sites Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\Zones\\\\2[\\s]*1406[\\s]*DWORD:1']), ('Windows Components\\Internet Explorer\\Internet Control Panel\\Security Page\\Locked-Down Trusted Sites Zone\\Access data sources across domains', {'Access data sources across domains': 'Prompt'}, ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Internet Settings\\\\lockdown_zones\\\\2[\\s]*1406[\\s]*DWORD:1'])])
def test_set_computer_policy(clean_comp, lgpo_bin, shell, name, setting, exp_regexes):
    if False:
        i = 10
        return i + 15
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name=name, setting=setting, exp_regexes=exp_regexes)

@pytest.mark.parametrize('name, setting, exp_regexes', [('Point and Print Restrictions', {'Users can only point and print to these servers': True, 'Enter fully qualified server names separated by semicolons': 'fakeserver1;fakeserver2', 'Users can only point and print to machines in their forest': True, 'When installing drivers for a new connection': 'Show warning and elevation prompt', 'When updating drivers for an existing connection': 'Show warning only'}, ['User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*Restricted[\\s]*DWORD:1', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*TrustedServers[\\s]*DWORD:1', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*ServerList[\\s]*SZ:fakeserver1;fakeserver2', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*InForest[\\s]*DWORD:1', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*NoWarningNoElevationOnInstall[\\s]*DWORD:0', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*UpdatePromptSettings[\\s]*DWORD:1']), ('Point and Print Restrictions', 'Disabled', ['User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*Restricted[\\s]*DWORD:0', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*TrustedServers[\\s]*DELETE', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*ServerList[\\s]*DELETE', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*InForest[\\s]*DELETE', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*NoWarningNoElevationOnInstall[\\s]*DELETE', 'User[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Printers\\\\PointAndPrint[\\s]*UpdatePromptSettings[\\s]*DELETE']), ('Point and Print Restrictions', 'Not Configured', ['; Source file:  c:\\\\windows\\\\system32\\\\grouppolicy\\\\user\\\\registry.pol[\\s]*; PARSING COMPLETED.'])])
def test_set_user_policy(clean_user, lgpo_bin, shell, name, setting, exp_regexes):
    if False:
        i = 10
        return i + 15
    _test_set_user_policy(lgpo_bin=lgpo_bin, shell=shell, name=name, setting=setting, exp_regexes=exp_regexes)

def test_set_computer_policy_windows_update(clean_comp, lgpo_bin, shell):
    if False:
        print('Hello World!')
    '\n    Test setting/unsetting/changing WindowsUpdate policy\n    '
    result = win_lgpo.get_policy_info(policy_name='Configure Automatic Updates', policy_class='Machine')
    the_policy = {}
    the_policy_check_enabled = ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*NoAutoUpdate[\\s]*DWORD:0']
    the_policy_check_disabled = ['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*NoAutoUpdate[\\s]*DWORD:1']
    for item in result['policy_elements']:
        if 'Configure automatic updating' in item['element_aliases']:
            the_policy.update({'Configure automatic updating': '4 - Auto download and schedule the install'})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AUOptions[\\s]*DWORD:4')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AUOptions[\\s]*DELETE')
        elif 'Install during automatic maintenance' in item['element_aliases']:
            the_policy.update({'Install during automatic maintenance': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AutomaticMaintenanceEnabled[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AutomaticMaintenanceEnabled[\\s]*DELETE')
        elif 'Scheduled install day' in item['element_aliases']:
            the_policy.update({'Scheduled install day': '7 - Every Saturday'})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallDay[\\s]*DWORD:7')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallDay[\\s]*DELETE')
        elif 'Scheduled install time' in item['element_aliases']:
            the_policy.update({'Scheduled install time': '17:00'})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallTime[\\s]*DWORD:17')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallTime[\\s]*DELETE')
        elif 'Install updates for other Microsoft products' in item['element_aliases']:
            the_policy.update({'Install updates for other Microsoft products': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AllowMUUpdateService[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AllowMUUpdateService[\\s]*DELETE')
        elif 'AutoUpdateSchEveryWeek' in item['element_aliases']:
            the_policy.update({'AutoUpdateSchEveryWeek': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallEveryWeek[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallEveryWeek[\\s]*DELETE')
        elif 'First week of the month' in item['element_aliases']:
            the_policy.update({'First week of the month': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallFirstWeek[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallFirstWeek[\\s]*DELETE')
        elif 'Second week of the month' in item['element_aliases']:
            the_policy.update({'Second week of the month': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallSecondWeek[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallSecondWeek[\\s]*DELETE')
        elif 'Third week of the month' in item['element_aliases']:
            the_policy.update({'Third week of the month': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallThirdWeek[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallThirdWeek[\\s]*DELETE')
        elif 'Fourth week of the month' in item['element_aliases']:
            the_policy.update({'Fourth week of the month': True})
            the_policy_check_enabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallFourthWeek[\\s]*DWORD:1\\s*')
            the_policy_check_disabled.append('Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallFourthWeek[\\s]*DELETE')
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='Configure Automatic Updates', setting=the_policy, exp_regexes=the_policy_check_enabled)
    clean_comp.unlink(missing_ok=True)
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='Configure Automatic Updates', setting='Disabled', exp_regexes=the_policy_check_disabled)
    clean_comp.unlink(missing_ok=True)
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='Configure Automatic Updates', setting='Not Configured', exp_regexes=['; Source file:  c:\\\\windows\\\\system32\\\\grouppolicy\\\\machine\\\\registry.pol[\\s]*; PARSING COMPLETED.'])

def test_set_computer_policy_multiple_policies(clean_comp, lgpo_bin, shell):
    if False:
        print('Hello World!')
    '\n    Tests setting several ADMX policies in succession and validating the\n    configuration\n    '
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='TS_CLIENT_CLIPBOARD', setting='Disabled', exp_regexes=['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fDisableClip[\\s]*DWORD:0'])
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='RA_Unsolicit', setting={'Permit remote control of this computer': 'Allow helpers to remotely control the computer', 'Helpers': ['administrators', 'user1']}, exp_regexes=['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fDisableClip[\\s]*DWORD:0', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*user1[\\s]*SZ:user1[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*administrators[\\s]*SZ:administrators[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicited[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicitedFullControl[\\s]*DWORD:1'])
    _test_set_computer_policy(lgpo_bin=lgpo_bin, shell=shell, name='Configure Automatic Updates', setting='Disabled', exp_regexes=['Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fDisableClip[\\s]*DWORD:0', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*user1[\\s]*SZ:user1[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services\\\\RAUnsolicit[\\s]*administrators[\\s]*SZ:administrators[\\s]*', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicited[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\policies\\\\Microsoft\\\\Windows NT\\\\Terminal Services[\\s]*fAllowUnsolicitedFullControl[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*NoAutoUpdate[\\s]*DWORD:1', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AUOptions[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AutomaticMaintenanceEnabled[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallDay[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*ScheduledInstallTime[\\s]*DELETE', 'Computer[\\s]*Software\\\\Policies\\\\Microsoft\\\\Windows\\\\WindowsUpdate\\\\AU[\\s]*AllowMUUpdateService[\\s]*DELETE'])