import os
import shutil
import pytest
from PyInstaller import HOMEPATH, PLATFORM
from PyInstaller.utils.tests import importorskip
from PyInstaller.utils.win32 import winmanifest

def _get_parent_tags(node):
    if False:
        for i in range(10):
            print('nop')
    tags = []
    while node is not None:
        tags.append(node.tag)
        node = node.getparent()
    return list(reversed(tags))

def _filter_whitespace_changes(actions):
    if False:
        i = 10
        return i + 15
    import xmldiff.actions
    filtered_actions = []
    for action in actions:
        if isinstance(action, (xmldiff.actions.UpdateTextAfter, xmldiff.actions.UpdateTextIn)):
            if not action.text.strip():
                continue
        filtered_actions.append(action)
    return filtered_actions
_REQUEST_EXECUTION_LEVEL_TAGS = ['{urn:schemas-microsoft-com:asm.v1}assembly', '{urn:schemas-microsoft-com:asm.v3}trustInfo', '{urn:schemas-microsoft-com:asm.v3}security', '{urn:schemas-microsoft-com:asm.v3}requestedPrivileges', '{urn:schemas-microsoft-com:asm.v3}requestedExecutionLevel']
_DEPENDENT_ASSEMBLY_IDENTITY_TAGS = ['{urn:schemas-microsoft-com:asm.v1}assembly', '{urn:schemas-microsoft-com:asm.v1}dependency', '{urn:schemas-microsoft-com:asm.v1}dependentAssembly', '{urn:schemas-microsoft-com:asm.v1}assemblyIdentity']
_MS_COMMON_CONTROLS_ATTRIBUTES = {'type': 'win32', 'name': 'Microsoft.Windows.Common-Controls', 'version': '6.0.0.0', 'processorArchitecture': '*', 'publicKeyToken': '6595b64144ccf1df', 'language': '*'}

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_default_manifest():
    if False:
        i = 10
        return i + 15
    import lxml
    import xmldiff.main
    app_manifest = winmanifest.create_application_manifest()
    tree_base = lxml.etree.fromstring(winmanifest._DEFAULT_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    assert not diff

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_default_manifest_uac_admin():
    if False:
        while True:
            i = 10
    import lxml
    import xmldiff.main
    import xmldiff.actions
    app_manifest = winmanifest.create_application_manifest(uac_admin=True)
    tree_base = lxml.etree.fromstring(winmanifest._DEFAULT_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    assert len(diff) == 1
    action = diff[0]
    assert isinstance(action, xmldiff.actions.UpdateAttrib)
    assert action.name == 'level'
    assert action.value == 'requireAdministrator'
    node = tree.xpath(action.node)[0]
    assert _get_parent_tags(node) == _REQUEST_EXECUTION_LEVEL_TAGS

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_default_manifest_uac_uiaccess():
    if False:
        i = 10
        return i + 15
    import lxml
    import xmldiff.main
    import xmldiff.actions
    app_manifest = winmanifest.create_application_manifest(uac_uiaccess=True)
    tree_base = lxml.etree.fromstring(winmanifest._DEFAULT_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    assert len(diff) == 1
    action = diff[0]
    assert isinstance(action, xmldiff.actions.UpdateAttrib)
    assert action.name == 'uiAccess'
    assert action.value == 'true'
    node = tree.xpath(action.node)[0]
    assert _get_parent_tags(node) == _REQUEST_EXECUTION_LEVEL_TAGS

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_default_manifest_uac_admin_and_uiaccess():
    if False:
        while True:
            i = 10
    import lxml
    import xmldiff.main
    import xmldiff.actions
    app_manifest = winmanifest.create_application_manifest(uac_admin=True, uac_uiaccess=True)
    tree_base = lxml.etree.fromstring(winmanifest._DEFAULT_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    assert len(diff) == 2
    action = diff[0]
    assert isinstance(action, xmldiff.actions.UpdateAttrib)
    assert action.name == 'level'
    assert action.value == 'requireAdministrator'
    node = tree.xpath(action.node)[0]
    assert _get_parent_tags(node) == _REQUEST_EXECUTION_LEVEL_TAGS
    action = diff[1]
    assert isinstance(action, xmldiff.actions.UpdateAttrib)
    assert action.name == 'uiAccess'
    assert action.value == 'true'
    node = tree.xpath(action.node)[0]
    assert _get_parent_tags(node) == _REQUEST_EXECUTION_LEVEL_TAGS

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_custom_manifest_no_trust_info():
    if False:
        print('Hello World!')
    import lxml
    import xmldiff.main
    import xmldiff.actions
    _MANIFEST_XML = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>\n    </application>\n  </compatibility>\n  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n    <windowsSettings>\n      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, unaware</dpiAwareness>\n    </windowsSettings>\n  </application>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n'
    app_manifest = winmanifest.create_application_manifest(_MANIFEST_XML)
    tree_base = lxml.etree.fromstring(_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    diff = _filter_whitespace_changes(diff)
    assert len(diff) == 6
    action = diff[0]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert tree.xpath(action.target)[0].tag == _REQUEST_EXECUTION_LEVEL_TAGS[0]
    assert action.tag == _REQUEST_EXECUTION_LEVEL_TAGS[1]
    action = diff[1]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _REQUEST_EXECUTION_LEVEL_TAGS[2]
    action = diff[2]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _REQUEST_EXECUTION_LEVEL_TAGS[3]
    action = diff[3]
    print(action)
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _REQUEST_EXECUTION_LEVEL_TAGS[4]
    action = diff[4]
    assert isinstance(action, xmldiff.actions.InsertAttrib)
    assert action.name == 'level'
    assert action.value == 'asInvoker'
    action = diff[5]
    assert isinstance(action, xmldiff.actions.InsertAttrib)
    assert action.name == 'uiAccess'
    assert action.value == 'false'

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_custom_manifest_no_ms_common_controls():
    if False:
        for i in range(10):
            print('nop')
    import lxml
    import xmldiff.main
    import xmldiff.actions
    _MANIFEST_XML = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>\n    </application>\n  </compatibility>\n  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n    <windowsSettings>\n      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, unaware</dpiAwareness>\n    </windowsSettings>\n  </application>\n</assembly>\n'
    app_manifest = winmanifest.create_application_manifest(_MANIFEST_XML)
    tree_base = lxml.etree.fromstring(_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    diff = _filter_whitespace_changes(diff)
    assert len(diff) == 9
    action = diff[0]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert tree.xpath(action.target)[0].tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[0]
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[1]
    action = diff[1]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[2]
    action = diff[2]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[3]
    added_attributes = {}
    for action in diff[3:]:
        assert isinstance(action, xmldiff.actions.InsertAttrib)
        added_attributes[action.name] = action.value
    assert added_attributes == _MS_COMMON_CONTROLS_ATTRIBUTES

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_custom_manifest_different_ms_common_controls():
    if False:
        print('Hello World!')
    import lxml
    import xmldiff.main
    import xmldiff.actions
    _MANIFEST_XML = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>\n    </application>\n  </compatibility>\n  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n    <windowsSettings>\n      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, unaware</dpiAwareness>\n    </windowsSettings>\n  </application>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="5.0.0.0"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n'
    app_manifest = winmanifest.create_application_manifest(_MANIFEST_XML)
    tree_base = lxml.etree.fromstring(_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    diff = _filter_whitespace_changes(diff)
    assert len(diff) == 4
    updated_attributes = {}
    added_attributes = {}
    for action in diff:
        if isinstance(action, xmldiff.actions.UpdateAttrib):
            updated_attributes[action.name] = action.value
        elif isinstance(action, xmldiff.actions.InsertAttrib):
            added_attributes[action.name] = action.value
        else:
            raise ValueError(f'Unexpected modification: {action}')
        assert tree.xpath(action.node)[0].tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[-1]
    assert updated_attributes == {'version': _MS_COMMON_CONTROLS_ATTRIBUTES['version']}
    assert added_attributes == {key: value for (key, value) in _MS_COMMON_CONTROLS_ATTRIBUTES.items() if key in ('processorArchitecture', 'publicKeyToken', 'language')}

@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_custom_manifest_no_ms_common_controls_with_custom_dependency():
    if False:
        while True:
            i = 10
    import lxml
    import xmldiff.main
    import xmldiff.actions
    _MANIFEST_XML = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>\n    </application>\n  </compatibility>\n  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n    <windowsSettings>\n      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2, unaware</dpiAwareness>\n    </windowsSettings>\n  </application>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="MyAwesomeLibrary" version="1.0.0.0"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n'
    app_manifest = winmanifest.create_application_manifest(_MANIFEST_XML)
    tree_base = lxml.etree.fromstring(_MANIFEST_XML)
    tree = lxml.etree.fromstring(app_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    diff = _filter_whitespace_changes(diff)
    assert len(diff) == 9
    action = diff[0]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert tree.xpath(action.target)[0].tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[0]
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[1]
    action = diff[1]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[2]
    action = diff[2]
    assert isinstance(action, xmldiff.actions.InsertNode)
    assert action.tag == _DEPENDENT_ASSEMBLY_IDENTITY_TAGS[3]
    added_attributes = {}
    for action in diff[3:]:
        assert isinstance(action, xmldiff.actions.InsertAttrib)
        added_attributes[action.name] = action.value
    assert added_attributes == _MS_COMMON_CONTROLS_ATTRIBUTES

@pytest.mark.win32
def test_manifest_write_to_exe(tmp_path):
    if False:
        print('Hello World!')
    bootloader_file = os.path.join(HOMEPATH, 'PyInstaller', 'bootloader', PLATFORM, 'run.exe')
    test_file = str(tmp_path / 'test_file.exe')
    shutil.copyfile(bootloader_file, test_file)
    app_manifest = winmanifest.create_application_manifest(uac_admin=False, uac_uiaccess=True)
    winmanifest.write_manifest_to_executable(test_file, app_manifest)
    read_manifest = winmanifest.read_manifest_from_executable(test_file)
    assert read_manifest == app_manifest

@pytest.mark.win32
@importorskip('lxml')
@importorskip('xmldiff')
def test_manifest_write_to_exe_non_ascii_characters(tmp_path):
    if False:
        print('Hello World!')
    import lxml
    import xmldiff.main
    import xmldiff.actions
    _MANIFEST_XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <assemblyIdentity name="日本語で書かれた名前" processorArchitecture="amd64" type="win32" version="1.0.0.0"/>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity language="*" name="Microsoft.Windows.Common-Controls" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" type="win32" version="6.0.0.0"/>\n    </dependentAssembly>\n  </dependency>\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"/>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"/>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"/>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"/>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>\n    </application>\n  </compatibility>\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n</assembly>\n'.encode('utf-8')
    bootloader_file = os.path.join(HOMEPATH, 'PyInstaller', 'bootloader', PLATFORM, 'run.exe')
    test_file = str(tmp_path / 'test_file.exe')
    shutil.copyfile(bootloader_file, test_file)
    app_manifest = winmanifest.create_application_manifest(_MANIFEST_XML, uac_admin=False, uac_uiaccess=False)
    winmanifest.write_manifest_to_executable(test_file, app_manifest)
    read_manifest = winmanifest.read_manifest_from_executable(test_file)
    assert read_manifest == app_manifest
    tree_base = lxml.etree.fromstring(_MANIFEST_XML)
    tree = lxml.etree.fromstring(read_manifest)
    diff = xmldiff.main.diff_trees(tree_base, tree)
    diff = _filter_whitespace_changes(diff)
    assert not diff