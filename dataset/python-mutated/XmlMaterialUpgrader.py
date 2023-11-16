import xml.etree.ElementTree as ET
from UM.VersionUpgrade import VersionUpgrade
from .XmlMaterialProfile import XmlMaterialProfile

class XmlMaterialUpgrader(VersionUpgrade):

    def getXmlVersion(self, serialized):
        if False:
            i = 10
            return i + 15
        return XmlMaterialProfile.getVersionFromSerialized(serialized)

    def _xmlVersionToSettingVersion(self, xml_version: str) -> int:
        if False:
            while True:
                i = 10
        return XmlMaterialProfile.xmlVersionToSettingVersion(xml_version)

    def upgradeMaterial(self, serialised, filename):
        if False:
            while True:
                i = 10
        data = ET.fromstring(serialised)
        metadata = data.iterfind('./um:metadata/*', {'um': 'http://www.ultimaker.com/material'})
        for entry in metadata:
            if _tag_without_namespace(entry) == 'version':
                entry.text = '2'
                break
        data.attrib['version'] = '1.3'
        new_serialised = ET.tostring(data, encoding='utf-8').decode('utf-8')
        return ([filename], [new_serialised])

def _tag_without_namespace(element):
    if False:
        print('Hello World!')
    return element.tag[element.tag.rfind('}') + 1:]