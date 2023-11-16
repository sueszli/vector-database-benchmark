import xml.dom
import xml.dom.minidom
RT_MANIFEST = 24
CREATEPROCESS_MANIFEST_RESOURCE_ID = 1
ISOLATIONAWARE_MANIFEST_RESOURCE_ID = 2
LANG_NEUTRAL = 0
_DEFAULT_MANIFEST_XML = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n    <security>\n      <requestedPrivileges>\n        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n      </requestedPrivileges>\n    </security>\n  </trustInfo>\n  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\n    <application>\n      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"></supportedOS>\n      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"></supportedOS>\n      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"></supportedOS>\n      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"></supportedOS>\n      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"></supportedOS>\n    </application>\n  </compatibility>\n  <application xmlns="urn:schemas-microsoft-com:asm.v3">\n    <windowsSettings>\n      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>\n    </windowsSettings>\n  </application>\n  <dependency>\n    <dependentAssembly>\n      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"></assemblyIdentity>\n    </dependentAssembly>\n  </dependency>\n</assembly>\n'

def _find_elements_by_tag(root, tag):
    if False:
        i = 10
        return i + 15
    '\n    Find all elements with given tag under the given root element.\n    '
    return [node for node in root.childNodes if node.nodeType == xml.dom.Node.ELEMENT_NODE and node.tagName == tag]

def _find_element_by_tag(root, tag):
    if False:
        return 10
    '\n    Attempt to find a single element with given tag under the given root element, and return None if no such element\n    is found. Raises an error if multiple elements are found.\n    '
    elements = _find_elements_by_tag(root, tag)
    if len(elements) > 1:
        raise ValueError(f'Expected a single {tag!r} element, found {len(elements)} element(s)!')
    if not elements:
        return None
    return elements[0]

def _set_execution_level(manifest_dom, root_element, uac_admin=False, uac_uiaccess=False):
    if False:
        print('Hello World!')
    '\n    Find <security> -> <requestedPrivileges> -> <requestedExecutionLevel> element, and set its `level` and `uiAccess`\n    attributes based on supplied arguments. Create the XML elements if necessary, as they are optional.\n    '
    trust_info_element = _find_element_by_tag(root_element, 'trustInfo')
    if not trust_info_element:
        trust_info_element = manifest_dom.createElement('trustInfo')
        trust_info_element.setAttribute('xmlns', 'urn:schemas-microsoft-com:asm.v3')
        root_element.appendChild(trust_info_element)
    security_element = _find_element_by_tag(trust_info_element, 'security')
    if not security_element:
        security_element = manifest_dom.createElement('security')
        trust_info_element.appendChild(security_element)
    requested_privileges_element = _find_element_by_tag(security_element, 'requestedPrivileges')
    if not requested_privileges_element:
        requested_privileges_element = manifest_dom.createElement('requestedPrivileges')
        security_element.appendChild(requested_privileges_element)
    requested_execution_level_element = _find_element_by_tag(requested_privileges_element, 'requestedExecutionLevel')
    if not requested_execution_level_element:
        requested_execution_level_element = manifest_dom.createElement('requestedExecutionLevel')
        requested_privileges_element.appendChild(requested_execution_level_element)
    requested_execution_level_element.setAttribute('level', 'requireAdministrator' if uac_admin else 'asInvoker')
    requested_execution_level_element.setAttribute('uiAccess', 'true' if uac_uiaccess else 'false')

def _ensure_common_controls_dependency(manifest_dom, root_element):
    if False:
        while True:
            i = 10
    '\n    Scan <dependency> elements for the one whose <<dependentAssembly> -> <assemblyIdentity> corresponds to the\n    `Microsoft.Windows.Common-Controls`. If found, overwrite its properties. If not, create new <dependency>\n    element with corresponding sub-elements and attributes.\n    '
    dependency_elements = _find_elements_by_tag(root_element, 'dependency')
    for dependency_element in dependency_elements:
        dependent_assembly_element = _find_element_by_tag(dependency_element, 'dependentAssembly')
        assembly_identity_element = _find_element_by_tag(dependent_assembly_element, 'assemblyIdentity')
        if assembly_identity_element.attributes['name'].value == 'Microsoft.Windows.Common-Controls':
            common_controls_element = assembly_identity_element
            break
    else:
        dependency_element = manifest_dom.createElement('dependency')
        root_element.appendChild(dependency_element)
        dependent_assembly_element = manifest_dom.createElement('dependentAssembly')
        dependency_element.appendChild(dependent_assembly_element)
        common_controls_element = manifest_dom.createElement('assemblyIdentity')
        dependent_assembly_element.appendChild(common_controls_element)
    common_controls_element.setAttribute('type', 'win32')
    common_controls_element.setAttribute('name', 'Microsoft.Windows.Common-Controls')
    common_controls_element.setAttribute('version', '6.0.0.0')
    common_controls_element.setAttribute('processorArchitecture', '*')
    common_controls_element.setAttribute('publicKeyToken', '6595b64144ccf1df')
    common_controls_element.setAttribute('language', '*')

def create_application_manifest(manifest_xml=None, uac_admin=False, uac_uiaccess=False):
    if False:
        return 10
    '\n    Create application manifest, from built-in or custom manifest XML template. If provided, `manifest_xml` must be\n    a string or byte string containing XML source. The returned manifest is a byte string, encoded in UTF-8.\n\n    This function sets the attributes of `requestedExecutionLevel` based on provided `uac_admin` and `auc_uiacces`\n    arguments (creating the parent elements in the XML, if necessary). It also scans `dependency` elements for the\n    entry corresponding to `Microsoft.Windows.Common-Controls` and creates or modifies it as necessary.\n    '
    if manifest_xml is None:
        manifest_xml = _DEFAULT_MANIFEST_XML
    with xml.dom.minidom.parseString(manifest_xml) as manifest_dom:
        root_element = manifest_dom.documentElement
        assert root_element.tagName == 'assembly'
        assert root_element.namespaceURI == 'urn:schemas-microsoft-com:asm.v1'
        assert root_element.attributes['manifestVersion'].value == '1.0'
        _set_execution_level(manifest_dom, root_element, uac_admin, uac_uiaccess)
        _ensure_common_controls_dependency(manifest_dom, root_element)
        output = manifest_dom.toprettyxml(indent='  ', encoding='UTF-8')
    output = [line for line in output.splitlines() if line.strip()]
    output[0] = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    output = b'\n'.join(output)
    return output

def write_manifest_to_executable(filename, manifest_xml):
    if False:
        i = 10
        return i + 15
    "\n    Write the given manifest XML to the given executable's RT_MANIFEST resource.\n    "
    from PyInstaller.utils.win32 import winresource
    names = [CREATEPROCESS_MANIFEST_RESOURCE_ID]
    languages = [LANG_NEUTRAL, '*']
    winresource.add_or_update_resource(filename, manifest_xml, RT_MANIFEST, names, languages)

def read_manifest_from_executable(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Read manifest from the given executable."\n    '
    from PyInstaller.utils.win32 import winresource
    resources = winresource.get_resources(filename, [RT_MANIFEST])
    if RT_MANIFEST not in resources:
        raise ValueError(f'No RT_MANIFEST resources found in {filename!r}.')
    resources = resources[RT_MANIFEST]
    if CREATEPROCESS_MANIFEST_RESOURCE_ID not in resources:
        raise ValueError(f'No RT_MANIFEST resource named CREATEPROCESS_MANIFEST_RESOURCE_ID found in {filename!r}.')
    resources = resources[CREATEPROCESS_MANIFEST_RESOURCE_ID]
    if LANG_NEUTRAL in resources:
        resources = resources[LANG_NEUTRAL]
    else:
        resources = next(iter(resources.items()))
    manifest_xml = resources
    return manifest_xml