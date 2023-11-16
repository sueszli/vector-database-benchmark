"""
Entity module tests
"""
import unittest
from txtai.pipeline import Entity

class TestEntity(unittest.TestCase):
    """
    Entity tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create entity instance.\n        '
        cls.entity = Entity('dslim/bert-base-NER')

    def testEntity(self):
        if False:
            print('Hello World!')
        '\n        Test entity\n        '
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg")
        self.assertEqual([e[0] for e in entities], ['Canada', 'Manhattan'])

    def testEntityFlatten(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test entity with flattened output\n        '
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", flatten=True)
        self.assertEqual(entities, ['Canada', 'Manhattan'])
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", flatten=True, join=True)
        self.assertEqual(entities, 'Canada Manhattan')

    def testEntityTypes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test entity type filtering\n        '
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", labels=['PER'])
        self.assertFalse(entities)