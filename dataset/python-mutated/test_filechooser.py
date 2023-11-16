from kivy.tests.common import GraphicUnitTest

class FileChooserTestCase(GraphicUnitTest):

    def test_filechooserlistview(self):
        if False:
            return 10
        from kivy.uix.filechooser import FileChooserListView
        from os.path import expanduser
        r = self.render
        wid = FileChooserListView(path=expanduser('~'))
        r(wid, 2)