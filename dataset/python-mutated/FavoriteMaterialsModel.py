from cura.Machines.Models.BaseMaterialsModel import BaseMaterialsModel
import cura.CuraApplication

class FavoriteMaterialsModel(BaseMaterialsModel):
    """Model that shows the list of favorite materials."""

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        cura.CuraApplication.CuraApplication.getInstance().getPreferences().preferenceChanged.connect(self._onFavoritesChanged)
        self._onChanged()

    def _onFavoritesChanged(self, preference_key: str) -> None:
        if False:
            print('Hello World!')
        'Triggered when any preference changes, but only handles it when the list of favourites is changed. '
        if preference_key != 'cura/favorite_materials':
            return
        self._onChanged()

    def _update(self):
        if False:
            return 10
        if not self._canUpdate():
            return
        super()._update()
        item_list = []
        for (root_material_id, container_node) in self._available_materials.items():
            if bool(container_node.getMetaDataEntry('removed', False)):
                continue
            if root_material_id not in self._favorite_ids:
                continue
            item = self._createMaterialItem(root_material_id, container_node)
            if item:
                item_list.append(item)
        item_list = sorted(item_list, key=lambda d: d['brand'].upper())
        self.setItems(item_list)