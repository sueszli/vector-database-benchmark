"""Import API packages test.

This is a Python test that verifies whether API v2 packages can be imported
from the current build or not.

It uses the `_api/v2/api_packages.txt` file from the local wheel file.
The `_api/v2/api_packages.txt` file is created during the process of generating
TensorFlow API v2 init files and is stored in the wheel file after the build.

See README.md file for "how to run" instruction.
"""
import logging
import unittest
import pkg_resources
logging.basicConfig(level=logging.INFO)

class ImportApiPackagesTest(unittest.TestCase):
    """ImportApiPackagesTest class. See description at the top of the file."""

    def setUp(self):
        if False:
            return 10

        def _get_api_packages_v2():
            if False:
                print('Hello World!')
            api_packages_path = pkg_resources.resource_filename('tensorflow', '_api/v2/api_packages.txt')
            logging.info('Load api packages file: %s', api_packages_path)
            with open(api_packages_path) as file:
                return set(file.read().splitlines())
        super().setUp()
        self.api_packages_v2 = _get_api_packages_v2()
        self.packages_for_skip = ['tensorflow.distribute.cluster_resolver', 'tensorflow.lite.experimental.authoring', 'tensorflow.distribute.experimental.coordinator', 'tensorflow.summary.experimental', 'tensorflow.distribute.coordinator', 'tensorflow.distribute.experimental.partitioners']

    def test_import_runtime(self):
        if False:
            print('Hello World!')
        'Try to import all packages from api packages file one by one.'
        version = 'v2'
        failed_packages = []
        logging.info('Try to import packages at runtime...')
        for package_name in self.api_packages_v2:
            short_package_name = package_name.replace(f'_api.{version}.', '')
            if short_package_name not in self.packages_for_skip:
                try:
                    __import__(short_package_name)
                except ImportError:
                    logging.exception('error importing %s', short_package_name)
                    failed_packages.append(package_name)
        if failed_packages:
            self.fail(f'Failed to import {len(failed_packages)}/{len(self.api_packages_v2)} packages {version}:\n{failed_packages}')
        logging.info('Import of packages was successful.')
if __name__ == '__main__':
    unittest.main()