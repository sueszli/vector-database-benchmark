import UM.Settings.Models.SettingVisibilityHandler

class MaterialSettingsVisibilityHandler(UM.Settings.Models.SettingVisibilityHandler.SettingVisibilityHandler):

    def __init__(self, parent=None, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, parent=parent, **kwargs)
        material_settings = {'default_material_print_temperature', 'default_material_bed_temperature', 'material_standby_temperature', 'cool_fan_speed', 'retraction_amount', 'retraction_speed'}
        self.setVisible(material_settings)