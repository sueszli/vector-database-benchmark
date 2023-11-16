from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtQml import QQmlEngine
from UM.Qt.ListModel import ListModel
from cura.Machines.Models.BaseMaterialsModel import BaseMaterialsModel

class MaterialTypesModel(ListModel):

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        QQmlEngine.setObjectOwnership(self, QQmlEngine.ObjectOwnership.CppOwnership)
        self.addRoleName(Qt.ItemDataRole.UserRole + 1, 'name')
        self.addRoleName(Qt.ItemDataRole.UserRole + 2, 'brand')
        self.addRoleName(Qt.ItemDataRole.UserRole + 3, 'colors')

class MaterialBrandsModel(BaseMaterialsModel):
    extruderPositionChanged = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        QQmlEngine.setObjectOwnership(self, QQmlEngine.ObjectOwnership.CppOwnership)
        self.addRoleName(Qt.ItemDataRole.UserRole + 1, 'name')
        self.addRoleName(Qt.ItemDataRole.UserRole + 2, 'material_types')
        self._update()

    def _update(self):
        if False:
            return 10
        if not self._canUpdate():
            return
        super()._update()
        brand_item_list = []
        brand_group_dict = {}
        for (root_material_id, container_node) in self._available_materials.items():
            if bool(container_node.getMetaDataEntry('removed', False)):
                continue
            if not bool(container_node.getMetaDataEntry('visible', True)):
                continue
            brand = container_node.getMetaDataEntry('brand', '')
            if brand.lower() == 'generic':
                continue
            if brand not in brand_group_dict:
                brand_group_dict[brand] = {}
            material_type = container_node.getMetaDataEntry('material', '')
            if material_type not in brand_group_dict[brand]:
                brand_group_dict[brand][material_type] = []
            item = self._createMaterialItem(root_material_id, container_node)
            if item:
                brand_group_dict[brand][material_type].append(item)
        for (brand, material_dict) in brand_group_dict.items():
            material_type_item_list = []
            brand_item = {'name': brand, 'material_types': MaterialTypesModel()}
            for (material_type, material_list) in material_dict.items():
                material_type_item = {'name': material_type, 'brand': brand, 'colors': BaseMaterialsModel()}
                material_list = sorted(material_list, key=lambda x: x['name'].upper())
                material_type_item['colors'].setItems(material_list)
                material_type_item_list.append(material_type_item)
            material_type_item_list = sorted(material_type_item_list, key=lambda x: x['name'].upper())
            brand_item['material_types'].setItems(material_type_item_list)
            brand_item_list.append(brand_item)
        brand_item_list = sorted(brand_item_list, key=lambda x: x['name'].upper())
        self.setItems(brand_item_list)