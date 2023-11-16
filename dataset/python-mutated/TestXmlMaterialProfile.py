from unittest.mock import patch, MagicMock
from UM.Qt.QtApplication import QtApplication
import pytest
import XmlMaterialProfile

def createXmlMaterialProfile(material_id):
    if False:
        i = 10
        return i + 15
    try:
        return XmlMaterialProfile.XmlMaterialProfile.XmlMaterialProfile(material_id)
    except AttributeError:
        return XmlMaterialProfile.XmlMaterialProfile(material_id)

def test_setName():
    if False:
        while True:
            i = 10
    material_1 = createXmlMaterialProfile('herpderp')
    material_2 = createXmlMaterialProfile('OMGZOMG')
    material_1.getMetaData()['base_file'] = 'herpderp'
    material_2.getMetaData()['base_file'] = 'herpderp'
    container_registry = MagicMock()
    container_registry.isReadOnly = MagicMock(return_value=False)
    container_registry.findInstanceContainers = MagicMock(return_value=[material_1, material_2])
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        material_1.setName('beep!')
    assert material_1.getName() == 'beep!'
    assert material_2.getName() == 'beep!'

def test_setDirty():
    if False:
        while True:
            i = 10
    material_1 = createXmlMaterialProfile('herpderp')
    material_2 = createXmlMaterialProfile('OMGZOMG')
    material_1.getMetaData()['base_file'] = 'herpderp'
    material_2.getMetaData()['base_file'] = 'herpderp'
    container_registry = MagicMock()
    container_registry.isReadOnly = MagicMock(return_value=False)
    container_registry.findContainers = MagicMock(return_value=[material_1, material_2])
    assert not material_1.isDirty()
    assert not material_2.isDirty()
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        material_2.setDirty(True)
    assert material_1.isDirty()
    assert material_2.isDirty()
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        material_1.setDirty(False)
    assert not material_1.isDirty()
    assert material_2.isDirty()

def test_serializeNonBaseMaterial():
    if False:
        i = 10
        return i + 15
    material_1 = createXmlMaterialProfile('herpderp')
    material_1.getMetaData()['base_file'] = 'omgzomg'
    container_registry = MagicMock()
    container_registry.isReadOnly = MagicMock(return_value=False)
    container_registry.findContainers = MagicMock(return_value=[material_1])
    with patch('UM.Settings.ContainerRegistry.ContainerRegistry.getInstance', MagicMock(return_value=container_registry)):
        with pytest.raises(NotImplementedError):
            material_1.serialize()