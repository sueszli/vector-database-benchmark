import unittest
from kivy.tests.common import GraphicUnitTest

def _build_rst():
    if False:
        while True:
            i = 10
    from kivy.uix.rst import RstDocument

    class _TestRstReplace(RstDocument):

        def __init__(self, **kwargs):
            if False:
                return 10
            super(_TestRstReplace, self).__init__(**kwargs)
            self.text = '\n    .. |uni| unicode:: 0xe4\n    .. |nbsp| unicode:: 0xA0\n    .. |text| replace:: is\n    .. |hop| replace:: replaced\n    .. _hop: https://kivy.org\n\n    |uni| |nbsp| |text| |hop|_\n    '
    return _TestRstReplace()

class RstSubstitutionTestCase(GraphicUnitTest):

    @unittest.skip('Currently segfault, but no idea why.')
    def test_rst_replace(self):
        if False:
            for i in range(10):
                print('nop')
        rst = _build_rst()
        self.render(rst)
        pg = rst.children[0].children[0].children[0]
        rendered_text = pg.text[:]
        compare_text = u'[color=202020ff][anchor=hop]ä \xa0 is [ref=None][color=ce5c00ff]replaced[/color][/ref][/color]'
        self.assertEqual(rendered_text, compare_text)
if __name__ == '__main__':
    import unittest
    unittest.main()