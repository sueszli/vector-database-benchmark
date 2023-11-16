from runner.koan import *

class AboutPackages(Koan):

    def test_subfolders_can_form_part_of_a_module_package(self):
        if False:
            print('Hello World!')
        from .a_package_folder.a_module import Duck
        duck = Duck()
        self.assertEqual(__, duck.name)

    def test_subfolders_become_modules_if_they_have_an_init_module(self):
        if False:
            i = 10
            return i + 15
        from .a_package_folder import an_attribute
        self.assertEqual(__, an_attribute)

    def test_use_absolute_imports_to_import_upper_level_modules(self):
        if False:
            while True:
                i = 10
        import contemplate_koans
        self.assertEqual(__, contemplate_koans.__name__)

    def test_import_a_module_in_a_subfolder_folder_using_an_absolute_path(self):
        if False:
            while True:
                i = 10
        from koans.a_package_folder.a_module import Duck
        self.assertEqual(__, Duck.__module__)