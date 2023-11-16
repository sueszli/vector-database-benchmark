from __future__ import absolute_import
import unittest2
import yaml
from st2common.util import spec_loader
from st2tests.fixtures.specs import __package__ as specs_fixture_package

class SpecLoaderTest(unittest2.TestCase):

    def test_spec_loader(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(isinstance(spec_loader.load_spec('st2common', 'openapi.yaml.j2'), dict))

    def test_bad_spec_duplicate_keys(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(yaml.constructor.ConstructorError, 'found duplicate key "swagger"', spec_loader.load_spec, specs_fixture_package, 'openapi.yaml.j2')