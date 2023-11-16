from django.test import TestCase
from wagtail.blocks.migrations.operations import AlterBlockValueOperation, ListChildrenToStructBlockOperation, RemoveStreamChildrenOperation, RemoveStructChildrenOperation, RenameStreamChildrenOperation, RenameStructChildrenOperation, StreamChildrenToListBlockOperation, StreamChildrenToStreamBlockOperation, StreamChildrenToStructBlockOperation
from wagtail.blocks.migrations.utils import apply_changes_to_raw_data
from wagtail.test.streamfield_migrations import factories, models

class FieldChildBlockTest(TestCase):
    """Tests involving changes to top level blocks"""

    def setUp(self):
        if False:
            print('Hello World!')
        raw_data = factories.SampleModelFactory(content__0__char1__value='Char Block 1', content__1__char2__value='Char Block 2', content__2__char1__value='Char Block 1', content__3__char2__value='Char Block 2').content.raw_data
        self.raw_data = raw_data

    def test_rename(self):
        if False:
            return 10
        'Rename `char1` blocks to `renamed1`\n\n        Check whether all `char1` blocks have been renamed correctly.\n        Check whether ids and values for renamed blocks are intact.\n        Check whether other block types are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=RenameStreamChildrenOperation(old_name='char1', new_name='renamed1'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[0]['type'], 'renamed1')
        self.assertEqual(altered_raw_data[2]['type'], 'renamed1')
        self.assertEqual(altered_raw_data[0]['id'], self.raw_data[0]['id'])
        self.assertEqual(altered_raw_data[2]['id'], self.raw_data[2]['id'])
        self.assertEqual(altered_raw_data[0]['value'], self.raw_data[0]['value'])
        self.assertEqual(altered_raw_data[2]['value'], self.raw_data[2]['value'])
        self.assertEqual(altered_raw_data[1], self.raw_data[1])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])

    def test_remove(self):
        if False:
            while True:
                i = 10
        'Remove all `char1` blocks\n\n        Check whether all `char1` blocks have been removed and whether other blocks are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=RemoveStreamChildrenOperation(name='char1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 2)
        self.assertEqual(altered_raw_data[0], self.raw_data[1])
        self.assertEqual(altered_raw_data[1], self.raw_data[3])

    def test_combine_to_listblock(self):
        if False:
            return 10
        'Combine all `char1` blocks into a new ListBlock named `list1`\n\n        Check whether no `char1` blocks are present among the stream children and whether other\n        blocks are intact.\n        Check whether a new `list1` block has been added to the stream children and whether it has\n        child blocks corresponding to the previous `char1` blocks.\n        Check whether the ids and values from the `char1` blocks are intact in the list children.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToListBlockOperation(block_name='char1', list_block_name='list1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 3)
        self.assertEqual(altered_raw_data[0], self.raw_data[1])
        self.assertEqual(altered_raw_data[1], self.raw_data[3])
        self.assertEqual(altered_raw_data[2]['type'], 'list1')
        self.assertEqual(len(altered_raw_data[2]['value']), 2)
        self.assertEqual(altered_raw_data[2]['value'][0]['type'], 'item')
        self.assertEqual(altered_raw_data[2]['value'][1]['type'], 'item')
        self.assertEqual(altered_raw_data[2]['value'][0]['id'], self.raw_data[0]['id'])
        self.assertEqual(altered_raw_data[2]['value'][1]['id'], self.raw_data[2]['id'])
        self.assertEqual(altered_raw_data[2]['value'][0]['value'], self.raw_data[0]['value'])
        self.assertEqual(altered_raw_data[2]['value'][1]['value'], self.raw_data[2]['value'])

    def test_combine_to_listblock_no_existing_children(self):
        if False:
            i = 10
            return i + 15
        'Combine all `simplestruct` blocks into a new ListBlock named `list1`\n        We have no `simplestruct` blocks in our existing data, so there should be no list1 blocks\n        created and the data should be intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToListBlockOperation(block_name='simplestruct', list_block_name='list1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 4)
        self.assertEqual(altered_raw_data[0], self.raw_data[0])
        self.assertEqual(altered_raw_data[1], self.raw_data[1])
        self.assertEqual(altered_raw_data[2], self.raw_data[2])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])

    def test_combine_single_type_to_streamblock(self):
        if False:
            while True:
                i = 10
        'Combine all `char1` blocks as children of a new StreamBlock named `stream1`\n\n        Check whether no `char1` blocks are present among the (top) stream children and whether\n        other blocks are intact.\n        Check whether a new `stream1` block has been added to the (top) stream children.\n        Check whether the new `stream1` block has the `char1` blocks as children and whether they\n        are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToStreamBlockOperation(block_names=['char1'], stream_block_name='stream1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 3)
        self.assertEqual(altered_raw_data[0], self.raw_data[1])
        self.assertEqual(altered_raw_data[1], self.raw_data[3])
        self.assertEqual(altered_raw_data[2]['type'], 'stream1')
        self.assertEqual(len(altered_raw_data[2]['value']), 2)
        self.assertEqual(altered_raw_data[2]['value'][0], self.raw_data[0])
        self.assertEqual(altered_raw_data[2]['value'][1], self.raw_data[2])

    def test_combine_multiple_types_to_streamblock(self):
        if False:
            print('Hello World!')
        'Combine all `char1` and `char2` blocks as children of a new StreamBlock named `stream1`\n\n        Check whether no `char1` or `char2` blocks are present among the (top) stream children.\n        Check whether a new `stream1` block has been added to the (top) stream children.\n        Check whether the new `stream1` block has the `char1` and `char2` blocks as children and\n        that they are intact.\n\n        Note:\n            We only have `char1` and `char2` blocks in our existing data.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToStreamBlockOperation(block_names=['char1', 'char2'], stream_block_name='stream1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 1)
        self.assertEqual(altered_raw_data[0]['type'], 'stream1')
        self.assertEqual(len(altered_raw_data[0]['value']), 4)
        self.assertEqual(altered_raw_data[0]['value'][0], self.raw_data[0])
        self.assertEqual(altered_raw_data[0]['value'][1], self.raw_data[1])
        self.assertEqual(altered_raw_data[0]['value'][2], self.raw_data[2])
        self.assertEqual(altered_raw_data[0]['value'][3], self.raw_data[3])

    def test_combine_to_streamblock_no_existing_children(self):
        if False:
            while True:
                i = 10
        'Combine all `simplestruct` blocks as children of a new StreamBlock named `stream1`\n        We have no `simplestruct` blocks in our existing data, so there should be no stream1 blocks\n        created and the data should be intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToStreamBlockOperation(block_names=['simplestruct'], stream_block_name='stream1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data), 4)
        self.assertEqual(altered_raw_data[0], self.raw_data[0])
        self.assertEqual(altered_raw_data[1], self.raw_data[1])
        self.assertEqual(altered_raw_data[2], self.raw_data[2])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])

    def test_to_structblock(self):
        if False:
            for i in range(10):
                print('nop')
        'Move each `char1` block inside a new StructBlock named `struct1`\n\n        Check whether each `char1` block has been replaced with a `struct1` block in the stream\n        children.\n        Check whether other blocks are intact.\n        Check whether each `struct1` block has a `char1` child and whether it has the value of the\n        previous `char1` block.\n\n        Note:\n            Block ids are not preserved here.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='', operation=StreamChildrenToStructBlockOperation('char1', 'struct1'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[0]['type'], 'struct1')
        self.assertEqual(altered_raw_data[2]['type'], 'struct1')
        self.assertEqual(altered_raw_data[1], self.raw_data[1])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])
        self.assertIn('char1', altered_raw_data[0]['value'])
        self.assertIn('char1', altered_raw_data[2]['value'])
        self.assertEqual(altered_raw_data[0]['value']['char1'], self.raw_data[0]['value'])
        self.assertEqual(altered_raw_data[2]['value']['char1'], self.raw_data[2]['value'])

    def test_alter_value(self):
        if False:
            i = 10
            return i + 15
        'Change the value of each `char1` block to `foo`\n\n        Check whether the value of each `char1` block has changed to `foo`.\n        Check whether the values of other blocks are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='char1', operation=AlterBlockValueOperation(new_value='foo'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[0]['value'], 'foo')
        self.assertEqual(altered_raw_data[1]['value'], self.raw_data[1]['value'])
        self.assertEqual(altered_raw_data[2]['value'], 'foo')
        self.assertEqual(altered_raw_data[3]['value'], self.raw_data[3]['value'])

class FieldStructChildBlockTest(TestCase):
    """Tests involving changes to direct children of a StructBlock

    We use `simplestruct` blocks as the StructBlocks here.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        raw_data = factories.SampleModelFactory(content__0__char1__value='Char Block 1', content__1='simplestruct', content__2='simplestruct', content__3__char2__value='Char Block 2').content.raw_data
        self.raw_data = raw_data

    def test_blocks_and_data_not_operated_on_intact(self):
        if False:
            return 10
        'Test whether other blocks and data not passed to an operation are intact.\n\n        We are checking whether the parts of the data which are not passed to an operation are\n        intact. Since the recursion process depends just on the block path and block structure,\n        this check is independent of the operation used. We will use a rename operation for now.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestruct', operation=RenameStructChildrenOperation(old_name='char1', new_name='renamed1'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[0], self.raw_data[0])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])
        self.assertEqual(altered_raw_data[1]['id'], self.raw_data[1]['id'])
        self.assertEqual(altered_raw_data[2]['id'], self.raw_data[2]['id'])
        self.assertEqual(altered_raw_data[1]['type'], self.raw_data[1]['type'])
        self.assertEqual(altered_raw_data[2]['type'], self.raw_data[2]['type'])

    def test_rename(self):
        if False:
            for i in range(10):
                print('nop')
        'Rename `simplestruct.char1` blocks to `renamed1`\n\n        Check whether all `simplestruct.char1` blocks have been renamed correctly.\n        Check whether values for renamed blocks are intact.\n        Check whether other children of `simplestruct` are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestruct', operation=RenameStructChildrenOperation(old_name='char1', new_name='renamed1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data[1]['value']), 2)
        self.assertEqual(len(altered_raw_data[2]['value']), 2)
        self.assertNotIn('char1', altered_raw_data[1]['value'])
        self.assertNotIn('char1', altered_raw_data[2]['value'])
        self.assertIn('renamed1', altered_raw_data[1]['value'])
        self.assertIn('renamed1', altered_raw_data[2]['value'])
        self.assertEqual(altered_raw_data[1]['value']['renamed1'], self.raw_data[1]['value']['char1'])
        self.assertEqual(altered_raw_data[2]['value']['renamed1'], self.raw_data[2]['value']['char1'])
        self.assertIn('char2', altered_raw_data[1]['value'])
        self.assertIn('char2', altered_raw_data[2]['value'])
        self.assertEqual(altered_raw_data[1]['value']['char2'], self.raw_data[1]['value']['char2'])
        self.assertEqual(altered_raw_data[2]['value']['char2'], self.raw_data[2]['value']['char2'])

    def test_remove(self):
        if False:
            i = 10
            return i + 15
        'Remove `simplestruct.char1` blocks\n\n        Check whether all `simplestruct.char1` blocks have been removed.\n        Check whether other children of `simplestruct` are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestruct', operation=RemoveStructChildrenOperation(name='char1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data[1]['value']), 1)
        self.assertEqual(len(altered_raw_data[2]['value']), 1)
        self.assertNotIn('char1', altered_raw_data[1]['value'])
        self.assertNotIn('char1', altered_raw_data[2]['value'])
        self.assertIn('char2', altered_raw_data[1]['value'])
        self.assertIn('char2', altered_raw_data[2]['value'])
        self.assertEqual(altered_raw_data[1]['value']['char2'], self.raw_data[1]['value']['char2'])
        self.assertEqual(altered_raw_data[2]['value']['char2'], self.raw_data[2]['value']['char2'])

class FieldStreamChildBlockTest(TestCase):
    """Tests involving changes to direct children of a StreamBlock

    We use `simplestream` blocks as the StreamBlocks here.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        raw_data = factories.SampleModelFactory(content__0__char1__value='Char Block 1', content__1='simplestream', content__1__simplestream__0__char1__value='Char Block 1', content__1__simplestream__1__char2__value='Char Block 2', content__1__simplestream__2__char1__value='Char Block 1', content__2='simplestream', content__2__simplestream__0__char1__value='Char Block 1', content__3__char2__value='Char Block 2').content.raw_data
        self.raw_data = raw_data

    def test_blocks_and_data_not_operated_on_intact(self):
        if False:
            i = 10
            return i + 15
        'Test whether other blocks and data not passed to an operation are intact.\n\n        We are checking whether the parts of the data which are not passed to an operation are\n        intact. Since the recursion process depends just on the block path and block structure,\n        this check is independent of the operation used. We will use a rename operation for now.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestream', operation=RenameStreamChildrenOperation(old_name='char1', new_name='renamed1'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[0], self.raw_data[0])
        self.assertEqual(altered_raw_data[3], self.raw_data[3])
        self.assertEqual(altered_raw_data[1]['id'], self.raw_data[1]['id'])
        self.assertEqual(altered_raw_data[2]['id'], self.raw_data[2]['id'])
        self.assertEqual(altered_raw_data[1]['type'], self.raw_data[1]['type'])
        self.assertEqual(altered_raw_data[2]['type'], self.raw_data[2]['type'])

    def test_rename(self):
        if False:
            i = 10
            return i + 15
        'Rename `simplestream.char1` blocks to `renamed1`\n\n        Check whether all `simplestream.char1` blocks have been renamed correctly.\n        Check whether values and ids for renamed blocks are intact.\n        Check whether other children of `simplestream` are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestream', operation=RenameStreamChildrenOperation(old_name='char1', new_name='renamed1'), streamfield=models.SampleModel.content)
        self.assertEqual(altered_raw_data[1]['value'][0]['type'], 'renamed1')
        self.assertEqual(altered_raw_data[1]['value'][2]['type'], 'renamed1')
        self.assertEqual(altered_raw_data[2]['value'][0]['type'], 'renamed1')
        self.assertEqual(altered_raw_data[1]['value'][0]['id'], self.raw_data[1]['value'][0]['id'])
        self.assertEqual(altered_raw_data[1]['value'][2]['id'], self.raw_data[1]['value'][2]['id'])
        self.assertEqual(altered_raw_data[2]['value'][0]['id'], self.raw_data[2]['value'][0]['id'])
        self.assertEqual(altered_raw_data[1]['value'][0]['value'], self.raw_data[1]['value'][0]['value'])
        self.assertEqual(altered_raw_data[1]['value'][2]['value'], self.raw_data[1]['value'][2]['value'])
        self.assertEqual(altered_raw_data[2]['value'][0]['value'], self.raw_data[2]['value'][0]['value'])
        self.assertEqual(altered_raw_data[1]['value'][1], self.raw_data[1]['value'][1])

    def test_remove(self):
        if False:
            while True:
                i = 10
        'Remove `simplestream.char1` blocks\n\n        Check whether all `simplestream.char1` blocks have been removed.\n        Check whether other children of `simplestream` are intact.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplestream', operation=RemoveStreamChildrenOperation(name='char1'), streamfield=models.SampleModel.content)
        self.assertEqual(len(altered_raw_data[1]['value']), 1)
        self.assertEqual(len(altered_raw_data[2]['value']), 0)
        self.assertEqual(altered_raw_data[1]['value'][0], self.raw_data[1]['value'][1])

class FieldListChildBlockTest(TestCase):
    """Tests involving changes to direct children of a ListBlock

    We use `simplelist` blocks as the ListBlocks here.
    """

    def setUp(self):
        if False:
            return 10
        raw_data = factories.SampleModelFactory(content__0__char1__value='Char Block 1', content__1__simplelist__0='Foo 1', content__1__simplelist__1='Foo 2', content__2__simplelist__0='Foo 3').content.raw_data
        self.raw_data = raw_data

    def test_to_structblock(self):
        if False:
            return 10
        'Turn each list child into a StructBlock and move value inside as a child named `text`\n\n        Check whether each list child has been converted to a StructBlock with a child named `text`\n        in it.\n        Check whether the previous value of each list child is now the value that `text` takes.\n\n        Note:\n            Block ids are not preserved here.\n        '
        altered_raw_data = apply_changes_to_raw_data(raw_data=self.raw_data, block_path_str='simplelist', operation=ListChildrenToStructBlockOperation(block_name='text'), streamfield=models.SampleModel.content)
        self.assertEqual(type(altered_raw_data[1]['value'][0]['value']), dict)
        self.assertEqual(type(altered_raw_data[1]['value'][1]['value']), dict)
        self.assertEqual(type(altered_raw_data[2]['value'][0]['value']), dict)
        self.assertIn('text', altered_raw_data[1]['value'][0]['value'])
        self.assertIn('text', altered_raw_data[1]['value'][1]['value'])
        self.assertIn('text', altered_raw_data[2]['value'][0]['value'])
        self.assertEqual(altered_raw_data[1]['value'][0]['value']['text'], self.raw_data[1]['value'][0]['value'])
        self.assertEqual(altered_raw_data[1]['value'][1]['value']['text'], self.raw_data[1]['value'][1]['value'])
        self.assertEqual(altered_raw_data[2]['value'][0]['value']['text'], self.raw_data[2]['value'][0]['value'])