from server.tests.utils import BaseTestCase

class TestZoneCompletions(BaseTestCase):

    def test_zones(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_zones("ui.form_card(box='')")
        self.assert_zones('ui.form_card(box="")')

    def test_zones_multilne(self):
        if False:
            print('Hello World!')
        self.assert_zones("ui.form_card(\nbox=''\n)", False)
        self.assert_zones('ui.form_card(\nbox=""\n)', False)

    def test_zones_box(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_zones("ui.box(zone='')")
        self.assert_zones('ui.box(zone="")')

    def test_zones_box_multiline(self):
        if False:
            print('Hello World!')
        self.assert_zones("ui.box(\nzone=''\n)", False)
        self.assert_zones('ui.box(\nzone=""\n)', False)

    def test_zones_box_positional(self):
        if False:
            i = 10
            return i + 15
        self.assert_zones("ui.box('')")
        self.assert_zones('ui.box("")')

    def test_zones_box_positional_multiline(self):
        if False:
            while True:
                i = 10
        self.assert_zones("ui.box(\n''\n)", False)
        self.assert_zones('ui.box(\n""\n)', False)

    def test_zones_box_str(self):
        if False:
            i = 10
            return i + 15
        self.assert_zones("ui.boxes('')")
        self.assert_zones('ui.boxes("")')

    def test_zones_box_str_multiple(self):
        if False:
            while True:
                i = 10
        self.assert_zones("ui.boxes('zone', '')")
        self.assert_zones("ui.boxes(\n''\n)", False)

    def test_zones_box_completes_only_zone(self):
        if False:
            return 10
        self.assertEqual(len(self.get_completions('ui.box("foo", width="")')), 0)
        self.assertEqual(len(self.get_completions('ui.box("foo", "")')), 0)