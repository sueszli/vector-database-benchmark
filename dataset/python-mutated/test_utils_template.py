import unittest
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from scrapy.utils.template import render_templatefile
__doctests__ = ['scrapy.utils.template']

class UtilsRenderTemplateFileTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tmp_path = mkdtemp()

    def tearDown(self):
        if False:
            return 10
        rmtree(self.tmp_path)

    def test_simple_render(self):
        if False:
            for i in range(10):
                print('nop')
        context = dict(project_name='proj', name='spi', classname='TheSpider')
        template = 'from ${project_name}.spiders.${name} import ${classname}'
        rendered = 'from proj.spiders.spi import TheSpider'
        template_path = Path(self.tmp_path, 'templ.py.tmpl')
        render_path = Path(self.tmp_path, 'templ.py')
        template_path.write_text(template, encoding='utf8')
        assert template_path.is_file()
        render_templatefile(template_path, **context)
        self.assertFalse(template_path.exists())
        self.assertEqual(render_path.read_text(encoding='utf8'), rendered)
        render_path.unlink()
        assert not render_path.exists()
if '__main__' == __name__:
    unittest.main()