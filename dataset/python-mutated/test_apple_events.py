"""
macOS-specific test to check handling of Apple Events in the bootloader.
"""
import json
import os
import subprocess
import time
import functools
import uuid
import pytest
from PyInstaller.compat import is_macos_11

def macos11_check_tmpdir(test):
    if False:
        while True:
            i = 10

    @functools.wraps(test)
    def wrapped(**kwargs):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = kwargs['tmpdir']
        if is_macos_11 and str(tmpdir).startswith(('/var', '/private/var')):
            pytest.skip('The custom URL schema registration does not work on macOS 11 when .app bundles are placed in the default temporary path.')
        return test(**kwargs)
    return wrapped

def _test_apple_events_handling(appname, tmpdir, pyi_builder_spec, monkeypatch, build_mode, argv_emu):
    if False:
        while True:
            i = 10

    def wait_for_event(logfile, event, timeout=60, polltime=0.25):
        if False:
            i = 10
            return i + 15
        "\n        Wait for the log file with 'started' or 'finished' entry to appear.\n        "
        assert event in {'started', 'finished'}, f'Invalid event: {event}!'
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed > timeout:
                return False
            if os.path.exists(logfile_path):
                with open(logfile_path) as fh:
                    log_lines = fh.readlines()
                    if log_lines:
                        if event == 'started':
                            first = log_lines[0]
                            assert first.startswith('started '), 'Unexpected line in log file!'
                            return True
                        elif log_lines[-1].startswith('finished '):
                            return True
            time.sleep(polltime)
    unique_key = int(time.time())
    custom_url_scheme = 'pyi-test-%i' % unique_key
    custom_file_ext = 'pyi_test_%i' % unique_key
    monkeypatch.setenv('PYI_CUSTOM_URL_SCHEME', custom_url_scheme)
    monkeypatch.setenv('PYI_CUSTOM_FILE_EXT', custom_file_ext)
    monkeypatch.setenv('PYI_BUILD_MODE', build_mode)
    monkeypatch.setenv('PYI_ARGV_EMU', str(int(argv_emu)))
    app_path = os.path.join(tmpdir, 'dist', appname + '.app')
    logfile_path = os.path.join(tmpdir, 'dist', 'events.log')
    pyi_builder_spec.test_spec(appname + '.spec', app_args=['5'])
    assert wait_for_event(logfile_path, 'started'), 'Timeout while waiting for app to start (test_spec run)!'
    assert wait_for_event(logfile_path, 'finished'), 'Timeout while waiting for app to finish (test_spec run)!'
    time.sleep(5)
    os.remove(logfile_path)
    old_dist = os.path.join(tmpdir, 'dist')
    new_dist = os.path.join(tmpdir, f'dist-{uuid.uuid4()}')
    os.rename(old_dist, new_dist)
    app_path = os.path.join(new_dist, appname + '.app')
    logfile_path = os.path.join(new_dist, 'events.log')
    subprocess.check_call(['open', app_path, '--args', '5'])
    assert wait_for_event(logfile_path, 'started'), 'Timeout while waiting for app to start (registration run)!'
    assert wait_for_event(logfile_path, 'finished'), 'Timeout while waiting for app to finish (registration run)!'
    time.sleep(5)
    os.remove(logfile_path)
    n_files = 32
    assoc_files = []
    for ii in range(n_files):
        assoc_path = os.path.join(tmpdir, 'AFile{}.{}'.format(ii, custom_file_ext))
        with open(assoc_path, 'wt') as fh:
            fh.write('File contents #{}\n'.format(ii))
        assoc_files.append(assoc_path)
    files_list = [('file://' if ii % 2 else '') + ff for (ii, ff) in enumerate(assoc_files)]
    subprocess.check_call(['open', *files_list])
    assert wait_for_event(logfile_path, 'started'), 'Timeout while waiting for app to start (test run)!'
    url_hello = custom_url_scheme + '://lowecase_required/hello_world/'
    subprocess.check_call(['open', url_hello])
    time.sleep(1.0)
    app_put_to_background = False
    try:
        subprocess.check_call(['osascript', '-e', 'tell application "System Events" to activate'])
        app_put_to_background = True
    except Exception:
        pass
    time.sleep(1.0)
    if app_put_to_background:
        subprocess.check_call(['open', app_path])
        time.sleep(1.0)
    files_list = [('file://' if ii % 2 else '') + ff for (ii, ff) in enumerate(assoc_files[:4])]
    subprocess.check_call(['open', *files_list])
    time.sleep(1.0)
    url_goodbye = custom_url_scheme + '://lowecase_required/goodybe_galaxy/'
    subprocess.check_call(['open', url_goodbye])
    time.sleep(1.0)
    url_large = custom_url_scheme + '://lowecase_required/large_data/'
    url_large += 'x' * 64000
    subprocess.check_call(['open', url_large])
    time.sleep(1.0)
    files_list = [('file://' if ii % 2 else '') + ff for (ii, ff) in enumerate(assoc_files[-4:])]
    subprocess.check_call(['open', *files_list])
    time.sleep(1.0)
    assert wait_for_event(logfile_path, 'finished'), 'Timeout while waiting for app to finish (test run)!'
    time.sleep(2)
    with open(logfile_path, 'r') as fh:
        log_lines = fh.readlines()
    assert log_lines[0].startswith('started '), 'Unexpected first line in log!'
    assert log_lines[-1].startswith('finished '), 'Unexpected last line in log!'
    events = []
    errors = []
    unknown = []
    for log_line in log_lines[1:-1]:
        if log_line.startswith('ae '):
            (_, event_id, event_data) = log_line.split(' ', 2)
            events.append((event_id, json.loads(event_data)))
        elif log_line.startswith('ERROR '):
            errors.append(log_line.split(' ', 1))
        else:
            unknown.append(log_line)
    assert not errors, 'Event log contains error(s)!'
    assert not unknown, 'Event log contains unknown line(s)!'
    data = json.loads(log_lines[0].split(' ', 1)[-1])
    args = data['args']
    data = json.loads(log_lines[-1].split(' ', 1)[-1])
    activation_count = data['activation_count']
    initial_oapp = True
    if build_mode == 'onedir' and (not argv_emu):
        initial_oapp = False
    event_idx = 0
    if initial_oapp:
        (event, data) = events[event_idx]
        event_idx += 1
        assert event == 'oapp'
    if argv_emu:
        assert args == assoc_files, 'Arguments received via argv-emu do not match expected list!'
    else:
        assert args == [], 'Application should receive no arguments when argv-emu is disabled!'
        (event, data) = events[event_idx]
        event_idx += 1
        assert event == 'odoc'
        assert data == assoc_files
    expected_activations = initial_oapp + app_put_to_background
    assert activation_count == expected_activations, 'Application did not handle activation event(s) as expected!'
    (event, data) = events[event_idx]
    event_idx += 1
    assert event == 'GURL'
    assert data == [url_hello]
    if app_put_to_background:
        (event, data) = events[event_idx]
        event_idx += 1
        assert event == 'rapp'
    (event, data) = events[event_idx]
    event_idx += 1
    assert event == 'odoc'
    assert data == assoc_files[:4]
    (event, data) = events[event_idx]
    event_idx += 1
    assert event == 'GURL'
    assert data == [url_goodbye]
    (event, data) = events[event_idx]
    event_idx += 1
    assert event == 'GURL'
    assert data == [url_large]
    (event, data) = events[event_idx]
    event_idx += 1
    assert event == 'odoc'
    assert data == assoc_files[-4:]
    for ff in assoc_files:
        try:
            os.remove(ff)
        except OSError:
            pass

@pytest.mark.darwin
@macos11_check_tmpdir
@pytest.mark.parametrize('build_mode', ['onefile', 'onedir'])
@pytest.mark.parametrize('argv_emu', [True, False], ids=['emu', 'noemu'])
def test_apple_event_handling_carbon(tmpdir, pyi_builder_spec, monkeypatch, build_mode, argv_emu):
    if False:
        i = 10
        return i + 15
    return _test_apple_events_handling('pyi_osx_aevent_handling_carbon', tmpdir, pyi_builder_spec, monkeypatch, build_mode, argv_emu)