from PyQt6.QtGui import QColor
from test.picardtestcase import PicardTestCase
from picard import config
from picard.ui.colors import InterfaceColors, UnknownColorException, interface_colors
settings = {'interface_colors': {'unknowncolor': '#deadbe', 'entity_error': '#abcdef'}, 'interface_colors_dark': {'unknowncolor': '#deadbe', 'entity_error': '#abcdef'}}

class InterfaceColorsTest(PicardTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.set_config_values(settings)

    def test_interface_colors(self):
        if False:
            return 10
        for key in ('interface_colors', 'interface_colors_dark'):
            interface_colors = InterfaceColors(dark_theme=key == 'interface_colors_dark')
            with self.assertRaises(UnknownColorException):
                interface_colors.get_color('testcolor')
            default_colors = interface_colors.default_colors
            self.assertEqual(interface_colors.get_color('entity_error'), default_colors['entity_error'].value)
            interface_colors.load_from_config()
            self.assertEqual(interface_colors.get_color('entity_error'), '#abcdef')
            self.assertEqual(interface_colors.get_colors()['entity_error'], '#abcdef')
            interface_colors.set_color('entity_error', '#000000')
            self.assertTrue(interface_colors.save_to_config())
            self.assertEqual(config.setting[key]['entity_error'], '#000000')
            self.assertNotIn('unknowncolor', config.setting[key])
            self.assertEqual(interface_colors.get_color_description('entity_error'), default_colors['entity_error'].description)
            self.assertEqual(interface_colors.get_qcolor('entity_error'), QColor('#000000'))

    def test_interface_colors_default(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(interface_colors, InterfaceColors)