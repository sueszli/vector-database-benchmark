from runner.koan import *
from .another_local_module import *
from .local_module_with_all_defined import *

class AboutModules(Koan):

    def test_importing_other_python_scripts_as_modules(self):
        if False:
            return 10
        from . import local_module
        duck = local_module.Duck()
        self.assertEqual(__, duck.name)

    def test_importing_attributes_from_classes_using_from_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        from .local_module import Duck
        duck = Duck()
        self.assertEqual(__, duck.name)

    def test_we_can_import_multiple_items_at_once(self):
        if False:
            for i in range(10):
                print('nop')
        from . import jims, joes
        jims_dog = jims.Dog()
        joes_dog = joes.Dog()
        self.assertEqual(__, jims_dog.identify())
        self.assertEqual(__, joes_dog.identify())

    def test_importing_all_module_attributes_at_once(self):
        if False:
            return 10
        '\n        importing all attributes at once is done like so:\n            from .another_local_module import *\n        The import wildcard cannot be used from within classes or functions.\n        '
        goose = Goose()
        hamster = Hamster()
        self.assertEqual(__, goose.name)
        self.assertEqual(__, hamster.name)

    def test_modules_hide_attributes_prefixed_by_underscores(self):
        if False:
            return 10
        with self.assertRaises(___):
            private_squirrel = _SecretSquirrel()

    def test_private_attributes_are_still_accessible_in_modules(self):
        if False:
            while True:
                i = 10
        from .local_module import Duck
        duck = Duck()
        self.assertEqual(__, duck._password)

    def test_a_module_can_limit_wildcard_imports(self):
        if False:
            return 10
        '\n        Examine results of:\n            from .local_module_with_all_defined import *\n        '
        goat = Goat()
        self.assertEqual(__, goat.name)
        lizard = _Velociraptor()
        self.assertEqual(__, lizard.name)
        with self.assertRaises(___):
            duck = SecretDuck()