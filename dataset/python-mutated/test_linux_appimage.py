"""
GNU/Linux-specific test to check the bootloader from the AppImage.
"""
import os
import pathlib
import stat
import subprocess
import pytest

@pytest.mark.linux
@pytest.mark.parametrize('arch', ['x86_64'])
def test_appimage_loading(tmp_path, pyi_builder_spec, arch):
    if False:
        return 10
    appimagetool = pathlib.Path.home() / ('appimagetool-%s.AppImage' % arch)
    if not appimagetool.is_file():
        pytest.skip('%s not found' % appimagetool)
    if not os.access(appimagetool, os.X_OK):
        st = appimagetool.stat()
        appimagetool.chmod(st.st_mode | stat.S_IXUSR)
    app_name = 'apptest'
    app_path = os.path.join(tmp_path, '%s-%s.AppImage' % (app_name, arch))
    pyi_builder_spec.test_source('print("OK")', app_name=app_name, pyi_args=['--onedir'])
    tools_dir = os.path.join(os.path.dirname(__file__), 'data', 'appimage')
    script = os.path.join(tools_dir, 'create.sh')
    subprocess.check_call(['bash', script, tools_dir, tmp_path, app_name])
    app_dir = os.path.join(tmp_path, 'dist', 'AppRun')
    subprocess.check_call([appimagetool, '--no-appstream', app_dir, app_path])
    st = os.stat(app_path)
    os.chmod(app_path, st.st_mode | stat.S_IXUSR)
    subprocess.check_call([app_path])