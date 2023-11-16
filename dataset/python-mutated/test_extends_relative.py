import os
from django.template import Context, Engine, TemplateSyntaxError
from django.test import SimpleTestCase
from .utils import ROOT
RELATIVE = os.path.join(ROOT, 'relative_templates')

class ExtendsRelativeBehaviorTests(SimpleTestCase):

    def test_normal_extend(self):
        if False:
            while True:
                i = 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('one.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one')

    def test_normal_extend_variable(self):
        if False:
            while True:
                i = 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('one_var.html')
        output = template.render(Context({'tmpl': './two.html'}))
        self.assertEqual(output.strip(), 'three two one')

    def test_dir1_extend(self):
        if False:
            return 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/one.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir1 one')

    def test_dir1_extend1(self):
        if False:
            return 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/one1.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir1 one')

    def test_dir1_extend2(self):
        if False:
            for i in range(10):
                print('nop')
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/one2.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir1 one')

    def test_dir1_extend3(self):
        if False:
            for i in range(10):
                print('nop')
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/one3.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir1 one')

    def test_dir2_extend(self):
        if False:
            return 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/dir2/one.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir2 one')

    def test_extend_error(self):
        if False:
            print('Hello World!')
        engine = Engine(dirs=[RELATIVE])
        msg = 'The relative path \'"./../two.html"\' points outside the file hierarchy that template \'error_extends.html\' is in.'
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            engine.render_to_string('error_extends.html')

class IncludeRelativeBehaviorTests(SimpleTestCase):

    def test_normal_include(self):
        if False:
            while True:
                i = 10
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/dir2/inc2.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'dir2 include')

    def test_normal_include_variable(self):
        if False:
            i = 10
            return i + 15
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/dir2/inc3.html')
        output = template.render(Context({'tmpl': './include_content.html'}))
        self.assertEqual(output.strip(), 'dir2 include')

    def test_dir2_include(self):
        if False:
            i = 10
            return i + 15
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/dir2/inc1.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three')

    def test_include_error(self):
        if False:
            while True:
                i = 10
        engine = Engine(dirs=[RELATIVE])
        msg = 'The relative path \'"./../three.html"\' points outside the file hierarchy that template \'error_include.html\' is in.'
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            engine.render_to_string('error_include.html')

class ExtendsMixedBehaviorTests(SimpleTestCase):

    def test_mixing1(self):
        if False:
            print('Hello World!')
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/two.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three two one dir2 one dir1 two')

    def test_mixing2(self):
        if False:
            print('Hello World!')
        engine = Engine(dirs=[RELATIVE])
        template = engine.get_template('dir1/three.html')
        output = template.render(Context({}))
        self.assertEqual(output.strip(), 'three dir1 three')

    def test_mixing_loop(self):
        if False:
            for i in range(10):
                print('nop')
        engine = Engine(dirs=[RELATIVE])
        msg = 'The relative path \'"./dir2/../looped.html"\' was translated to template name \'dir1/looped.html\', the same template in which the tag appears.'
        with self.assertRaisesMessage(TemplateSyntaxError, msg):
            engine.render_to_string('dir1/looped.html')