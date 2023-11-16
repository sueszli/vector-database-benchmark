"""Tests for the 'parentwork' plugin."""
import os
import unittest
from test.helper import TestHelper
from unittest.mock import patch
from beets.library import Item
from beetsplug import parentwork
work = {'work': {'id': '1', 'title': 'work', 'work-relation-list': [{'type': 'parts', 'direction': 'backward', 'work': {'id': '2'}}], 'artist-relation-list': [{'type': 'composer', 'artist': {'name': 'random composer', 'sort-name': 'composer, random'}}]}}
dp_work = {'work': {'id': '2', 'title': 'directparentwork', 'work-relation-list': [{'type': 'parts', 'direction': 'backward', 'work': {'id': '3'}}], 'artist-relation-list': [{'type': 'composer', 'artist': {'name': 'random composer', 'sort-name': 'composer, random'}}]}}
p_work = {'work': {'id': '3', 'title': 'parentwork', 'artist-relation-list': [{'type': 'composer', 'artist': {'name': 'random composer', 'sort-name': 'composer, random'}}]}}

def mock_workid_response(mbid, includes):
    if False:
        return 10
    if mbid == '1':
        return work
    elif mbid == '2':
        return dp_work
    elif mbid == '3':
        return p_work

class ParentWorkIntegrationTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            print('Hello World!')
        'Set up configuration'
        self.setup_beets()
        self.load_plugins('parentwork')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.unload_plugins()
        self.teardown_beets()

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_normal_case_real(self):
        if False:
            while True:
                i = 10
        item = Item(path='/file', mb_workid='e27bda6e-531e-36d3-9cd7-b8ebc18e8c53', parentwork_workid_current='e27bda6e-531e-36d3-9cd7-                    b8ebc18e8c53')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], '32c8943f-1b27-3a23-8660-4567f4847c94')

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_force_real(self):
        if False:
            return 10
        self.config['parentwork']['force'] = True
        item = Item(path='/file', mb_workid='e27bda6e-531e-36d3-9cd7-b8ebc18e8c53', mb_parentworkid='XXX', parentwork_workid_current='e27bda6e-531e-36d3-9cd7-                    b8ebc18e8c53', parentwork='whatever')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], '32c8943f-1b27-3a23-8660-4567f4847c94')

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_no_force_real(self):
        if False:
            i = 10
            return i + 15
        self.config['parentwork']['force'] = False
        item = Item(path='/file', mb_workid='e27bda6e-531e-36d3-9cd7-                    b8ebc18e8c53', mb_parentworkid='XXX', parentwork_workid_current='e27bda6e-531e-36d3-9cd7-                    b8ebc18e8c53', parentwork='whatever')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], 'XXX')

    @unittest.skipUnless(os.environ.get('INTEGRATION_TEST', '0') == '1', 'integration testing not enabled')
    def test_direct_parent_work_real(self):
        if False:
            print('Hello World!')
        mb_workid = '2e4a3668-458d-3b2a-8be2-0b08e0d8243a'
        self.assertEqual('f04b42df-7251-4d86-a5ee-67cfa49580d1', parentwork.direct_parent_id(mb_workid)[0])
        self.assertEqual('45afb3b2-18ac-4187-bc72-beb1b1c194ba', parentwork.work_parent_id(mb_workid)[0])

class ParentWorkTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Set up configuration'
        self.setup_beets()
        self.load_plugins('parentwork')
        self.patcher = patch('musicbrainzngs.get_work_by_id', side_effect=mock_workid_response)
        self.patcher.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.unload_plugins()
        self.teardown_beets()
        self.patcher.stop()

    def test_normal_case(self):
        if False:
            i = 10
            return i + 15
        item = Item(path='/file', mb_workid='1', parentwork_workid_current='1')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], '3')

    def test_force(self):
        if False:
            i = 10
            return i + 15
        self.config['parentwork']['force'] = True
        item = Item(path='/file', mb_workid='1', mb_parentworkid='XXX', parentwork_workid_current='1', parentwork='parentwork')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], '3')

    def test_no_force(self):
        if False:
            for i in range(10):
                print('nop')
        self.config['parentwork']['force'] = False
        item = Item(path='/file', mb_workid='1', mb_parentworkid='XXX', parentwork_workid_current='1', parentwork='parentwork')
        item.add(self.lib)
        self.run_command('parentwork')
        item.load()
        self.assertEqual(item['mb_parentworkid'], 'XXX')

    def test_direct_parent_work(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('2', parentwork.direct_parent_id('1')[0])
        self.assertEqual('3', parentwork.work_parent_id('1')[0])

def suite():
    if False:
        return 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')