import pytest
from unittest import TestCase

class TestDatasetsImport(TestCase):

    def test_datasets_replace(self):
        if False:
            i = 10
            return i + 15
        from torchvision import datasets
        origin_set = set(datasets.__all__)
        del datasets
        from bigdl.nano.pytorch.vision import datasets
        new_set = set(dir(datasets))
        assert origin_set.issubset(new_set)

    def test_datasets_ImageFolder_version(self):
        if False:
            while True:
                i = 10
        from bigdl.nano.pytorch.vision import datasets
        assert datasets.__name__ in datasets.ImageFolder.__module__
if __name__ == '__main__':
    pytest.main([__file__])