from cura.Machines.Models.BaseMaterialsModel import BaseMaterialsModel

class GenericMaterialsModel(BaseMaterialsModel):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._onChanged()

    def _update(self):
        if False:
            i = 10
            return i + 15
        if not self._canUpdate():
            return
        super()._update()
        item_list = []
        for (root_material_id, container_node) in self._available_materials.items():
            if bool(container_node.getMetaDataEntry('removed', False)):
                continue
            if container_node.getMetaDataEntry('brand', 'unknown').lower() != 'generic':
                continue
            item = self._createMaterialItem(root_material_id, container_node)
            if item:
                item_list.append(item)
        item_list = sorted(item_list, key=lambda d: d['name'].upper())
        self.setItems(item_list)