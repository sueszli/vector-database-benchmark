"""
FiftyOne synchronization-related unit tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import unittest
import fiftyone as fo
from decorators import drop_datasets

class SingleProcessSynchronizationTests(unittest.TestCase):
    """Tests ensuring that when a dataset or samples in a dataset are modified
    all relevant objects are instantly in sync within the same process.
    """

    @drop_datasets
    def test_dataset_singleton(self):
        if False:
            print('Hello World!')
        'Tests that datasets are singletons.'
        dataset1 = fo.Dataset('test_dataset')
        dataset2 = fo.load_dataset('test_dataset')
        dataset3 = fo.Dataset()
        self.assertIs(dataset1, dataset2)
        self.assertIsNot(dataset1, dataset3)
        with self.assertRaises(ValueError):
            fo.Dataset('test_dataset')
        dataset1.delete()
        new_dataset1 = fo.Dataset('test_dataset')
        dataset1.__class__._instances['test_dataset'] = dataset1
        new_dataset1 = fo.load_dataset('test_dataset')
        self.assertIsNot(dataset1, new_dataset1)

    @drop_datasets
    def test_sample_singletons(self):
        if False:
            return 10
        'Tests that samples are singletons.'
        dataset = fo.Dataset()
        filepath = 'test1.png'
        sample = fo.Sample(filepath=filepath)
        dataset.add_sample(sample)
        sample2 = dataset[sample.id]
        self.assertIs(sample2, sample)
        sample3 = fo.Sample(filepath='test2.png')
        dataset.add_sample(sample3)
        self.assertIsNot(sample3, sample)
        sample4 = dataset.match({'filepath': os.path.abspath(filepath)}).first()
        self.assertIsNot(sample4, sample)

    @drop_datasets
    def test_dataset_add_delete_field(self):
        if False:
            while True:
                i = 10
        'Tests that when fields are added or removed from a dataset field\n        schema, those changes are reflected on the samples in the dataset.\n        '
        dataset = fo.Dataset()
        sample = fo.Sample(filepath='test1.png')
        dataset.add_sample(sample)
        field_name = 'field1'
        ftype = fo.IntField
        with self.assertRaises(AttributeError):
            sample.get_field(field_name)
        with self.assertRaises(KeyError):
            sample[field_name]
        with self.assertRaises(AttributeError):
            getattr(sample, field_name)
        dataset.add_sample_field(field_name, ftype)
        self.assertIsNone(sample.get_field(field_name))
        self.assertIsNone(sample[field_name])
        self.assertIsNone(getattr(sample, field_name))
        dataset.delete_sample_field(field_name)
        with self.assertRaises(AttributeError):
            sample.get_field(field_name)
        with self.assertRaises(KeyError):
            sample[field_name]
        with self.assertRaises(AttributeError):
            getattr(sample, field_name)
        value = 51
        dataset.add_sample_field(field_name, ftype)
        sample[field_name] = value
        self.assertEqual(sample.get_field(field_name), value)
        self.assertEqual(sample[field_name], value)
        self.assertEqual(getattr(sample, field_name), value)
        dataset.delete_sample_field(field_name)
        with self.assertRaises(AttributeError):
            sample.get_field(field_name)
        with self.assertRaises(KeyError):
            sample[field_name]
        with self.assertRaises(AttributeError):
            getattr(sample, field_name)

    @drop_datasets
    def test_dataset_delete_samples(self):
        if False:
            return 10
        'Tests that when a sample is deleted from a dataset, the sample is\n        disconnected from the dataset.\n        '
        dataset = fo.Dataset()
        sample = fo.Sample(filepath='test1.png')
        dataset.add_sample(sample)
        self.assertTrue(sample.in_dataset)
        self.assertIsNotNone(sample.id)
        self.assertIs(sample.dataset, dataset)
        dataset.delete_samples(sample)
        self.assertFalse(sample.in_dataset)
        self.assertIsNone(sample.id)
        self.assertIsNone(sample.dataset)
        filepath_template = 'test%d.png'
        num_samples = 10
        samples = [fo.Sample(filepath=filepath_template % i) for i in range(num_samples)]
        dataset.add_samples(samples)
        for sample in samples:
            self.assertTrue(sample.in_dataset)
            self.assertIsNotNone(sample.id)
            self.assertIs(sample.dataset, dataset)
        num_delete = 7
        dataset.delete_samples([sample.id for sample in samples[:num_delete]])
        for (i, sample) in enumerate(samples):
            if i < num_delete:
                self.assertFalse(sample.in_dataset)
                self.assertIsNone(sample.id)
                self.assertIsNone(sample.dataset)
            else:
                self.assertTrue(sample.in_dataset)
                self.assertIsNotNone(sample.id)
                self.assertIs(sample.dataset, dataset)
        dataset.clear()
        for sample in samples:
            self.assertFalse(sample.in_dataset)
            self.assertIsNone(sample.id)
            self.assertIsNone(sample.dataset)

    @drop_datasets
    def test_sample_set_field(self):
        if False:
            while True:
                i = 10
        'Tests that when a field is added to the dataset schema via implicit\n        adding on a sample, that change is reflected in the dataset.\n        '
        dataset = fo.Dataset()
        sample = fo.Sample(filepath='test1.png')
        dataset.add_sample(sample)
        field_name = 'field1'
        ftype = fo.IntField
        value = 51
        with self.assertRaises(KeyError):
            fields = dataset.get_field_schema()
            fields[field_name]
        sample[field_name] = value
        fields = dataset.get_field_schema()
        self.assertIsInstance(fields[field_name], ftype)

class ScopedObjectsSynchronizationTests(unittest.TestCase):
    """Tests ensuring that when a dataset or samples in a dataset are modified,
    those changes are passed on the the database and can be seen in different
    scopes (or processes!).
    """

    @drop_datasets
    def test_dataset(self):
        if False:
            i = 10
            return i + 15
        dataset_name = self.test_dataset.__name__

        def create_dataset():
            if False:
                i = 10
                return i + 15
            with self.assertRaises(ValueError):
                dataset = fo.load_dataset(dataset_name)
            dataset = fo.Dataset(dataset_name)
        create_dataset()

        def check_create_dataset():
            if False:
                while True:
                    i = 10
            fo.load_dataset(dataset_name)

        def check_create_dataset_via_load():
            if False:
                while True:
                    i = 10
            self.assertIn(dataset_name, fo.list_datasets())
            dataset = fo.load_dataset(dataset_name)
        check_create_dataset()
        check_create_dataset_via_load()

        def delete_default_field():
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            with self.assertRaises(ValueError):
                dataset.delete_sample_field('tags')
        delete_default_field()
        field_name = 'test_field'
        ftype = fo.IntField

        def add_field():
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            dataset.add_sample_field(field_name, ftype)

        def check_add_field():
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            fields = dataset.get_field_schema()
            self.assertIn(field_name, fields)
            self.assertIsInstance(fields[field_name], ftype)
        add_field()
        check_add_field()

        def delete_field():
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            dataset.delete_sample_field(field_name)

        def check_delete_field():
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            fields = dataset.get_field_schema()
            with self.assertRaises(KeyError):
                fields[field_name]
            sample_fields = dataset._doc.sample_fields
            sample_field_names = [sf.name for sf in sample_fields]
            self.assertNotIn(field_name, sample_field_names)
        delete_field()
        check_delete_field()

        def delete_dataset():
            if False:
                while True:
                    i = 10
            fo.delete_dataset(dataset_name)
        delete_dataset()

        def check_delete_dataset():
            if False:
                return 10
            with self.assertRaises(ValueError):
                fo.load_dataset(dataset_name)
        check_delete_dataset()

    @drop_datasets
    def test_add_delete_sample(self):
        if False:
            while True:
                i = 10
        dataset_name = self.test_add_delete_sample.__name__

        def create_dataset():
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.Dataset(dataset_name)
        create_dataset()
        filepath = 'test1.png'

        def add_sample():
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = fo.Sample(filepath=filepath)
            return dataset.add_sample(sample)

        def check_add_sample(sample_id):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            self.assertEqual(len(dataset), 1)
            sample = dataset[sample_id]
            self.assertTrue(sample.in_dataset)
            self.assertIsNotNone(sample.id)
            self.assertIs(sample.dataset, dataset)
        sample_id = add_sample()
        check_add_sample(sample_id)

        def delete_sample(sample_id):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            dataset.delete_samples(sample)

        def check_delete_sample(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            self.assertEqual(len(dataset), 0)
            with self.assertRaises(KeyError):
                dataset[sample_id]
        delete_sample(sample_id)
        check_delete_sample(sample_id)
        filepath_template = 'test_multi%d.png'
        num_samples = 10

        def add_samples():
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            samples = [fo.Sample(filepath=filepath_template % i) for i in range(num_samples)]
            return dataset.add_samples(samples)

        def check_add_samples(sample_ids):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            self.assertEqual(len(dataset), num_samples)
            for sample_id in sample_ids:
                sample = dataset[sample_id]
                self.assertTrue(sample.in_dataset)
                self.assertIsNotNone(sample.id)
                self.assertIs(sample.dataset, dataset)
        sample_ids = add_samples()
        check_add_samples(sample_ids)
        num_delete = 7

        def delete_samples(sample_ids):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            dataset.delete_samples(sample_ids[:num_delete])

        def check_delete_samples(sample_ids):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            self.assertEqual(len(dataset), num_samples - num_delete)
            for (i, sample_id) in enumerate(sample_ids):
                if i < num_delete:
                    with self.assertRaises(KeyError):
                        dataset[sample_id]
                else:
                    sample = dataset[sample_id]
                    self.assertTrue(sample.in_dataset)
                    self.assertIsNotNone(sample.id)
                    self.assertIs(sample.dataset, dataset)
        delete_samples(sample_ids)
        check_delete_samples(sample_ids)

        def clear_dataset():
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            dataset.clear()

        def check_clear_dataset(sample_ids):
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            self.assertEqual(len(dataset), 0)
            for sample_id in sample_ids:
                with self.assertRaises(KeyError):
                    dataset[sample_id]
        clear_dataset()
        check_clear_dataset(sample_ids)

    @drop_datasets
    def test_add_sample_expand_schema(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_name = self.test_add_sample_expand_schema.__name__

        def create_dataset():
            if False:
                while True:
                    i = 10
            dataset = fo.Dataset(dataset_name)
        create_dataset()

        def add_sample_expand_schema():
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = fo.Sample(filepath='test.png', test_field=True)
            return dataset.add_sample(sample)

        def check_add_sample_expand_schema(sample_id):
            if False:
                i = 10
                return i + 15
            dataset = fo.load_dataset(dataset_name)
            fields = dataset.get_field_schema()
            self.assertIn('test_field', fields)
            self.assertIsInstance(fields['test_field'], fo.BooleanField)
            sample = dataset[sample_id]
            self.assertEqual(sample['test_field'], True)
        sample_id = add_sample_expand_schema()
        check_add_sample_expand_schema(sample_id)

        def add_samples_expand_schema():
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample1 = fo.Sample(filepath='test1.png', test_field_1=51)
            sample2 = fo.Sample(filepath='test2.png', test_field_2='fiftyone')
            return dataset.add_samples([sample1, sample2])

        def check_add_samples_expand_schema(sample_ids):
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            fields = dataset.get_field_schema()
            self.assertIn('test_field_1', fields)
            self.assertIsInstance(fields['test_field_1'], fo.IntField)
            self.assertIn('test_field_2', fields)
            self.assertIsInstance(fields['test_field_2'], fo.StringField)
            sample1 = dataset[sample_ids[0]]
            self.assertEqual(sample1['test_field_1'], 51)
            sample2 = dataset[sample_ids[1]]
            self.assertEqual(sample2['test_field_2'], 'fiftyone')
        sample_ids = add_samples_expand_schema()
        check_add_samples_expand_schema(sample_ids)

    @drop_datasets
    def test_set_field_create(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_name = self.test_set_field_create.__name__

        def create_dataset():
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.Dataset(dataset_name)
            sample = fo.Sample(filepath='/path/to/image.jpg')
            return dataset.add_sample(sample)
        sample_id = create_dataset()
        field_name = 'field2'
        value = 51

        def set_field_create(sample_id):
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample[field_name] = value
            sample.save()

        def check_set_field_create(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            fields = dataset.get_field_schema()
            self.assertIn(field_name, fields)
            sample = dataset[sample_id]
            self.assertEqual(sample[field_name], value)
        set_field_create(sample_id)
        check_set_field_create(sample_id)

    @drop_datasets
    def test_modify_sample(self):
        if False:
            i = 10
            return i + 15
        dataset_name = self.test_set_field_create.__name__

        def create_dataset():
            if False:
                while True:
                    i = 10
            dataset = fo.Dataset(dataset_name)
            dataset.add_sample_field('bool_field', fo.BooleanField)
            dataset.add_sample_field('list_field', fo.ListField)
            return dataset.add_sample(fo.Sample(filepath='test.png'))
        sample_id = create_dataset()

        def check_field_defaults(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertIs(sample.bool_field, None)
            self.assertEqual(sample.list_field, None)
        check_field_defaults(sample_id)

        def modify_simple_field(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.bool_field = True
            sample.save()

        def check_modify_simple_field(sample_id):
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertIs(sample.bool_field, True)
        modify_simple_field(sample_id)
        check_modify_simple_field(sample_id)

        def clear_simple_field(sample_id):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            del sample.bool_field
            sample.save()

        def check_clear_simple_field(sample_id):
            if False:
                i = 10
                return i + 15
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertIs(sample.bool_field, None)
        clear_simple_field(sample_id)
        check_clear_simple_field(sample_id)

        def modify_list_set(sample_id):
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field = [True, False, True]
            sample.save()

        def check_modify_list_set(sample_id):
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertListEqual(sample.list_field, [True, False, True])
        modify_list_set(sample_id)
        check_modify_list_set(sample_id)

        def clear_complex_field(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field = None
            sample.save()

        def check_clear_complex_field(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertEqual(sample.list_field, None)
        clear_complex_field(sample_id)
        check_clear_complex_field(sample_id)

        def modify_list_set_again(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field = [51]
            sample.save()

        def check_modify_list_set_agin(sample_id):
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertListEqual(sample.list_field, [51])
        modify_list_set_again(sample_id)
        check_modify_list_set_agin(sample_id)

        def modify_list_extend(sample_id):
            if False:
                i = 10
                return i + 15
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field.extend(['fiftyone'])
            sample.save()

        def check_modify_list_extend(sample_id):
            if False:
                print('Hello World!')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertListEqual(sample.list_field, [51, 'fiftyone'])
        modify_list_extend(sample_id)
        check_modify_list_extend(sample_id)

        def modify_list_pop(sample_id):
            if False:
                return 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field.pop(0)
            sample.save()

        def check_modify_list_pop(sample_id):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertListEqual(sample.list_field, ['fiftyone'])
        modify_list_pop(sample_id)
        check_modify_list_pop(sample_id)

        def modify_list_iadd(sample_id):
            if False:
                for i in range(10):
                    print('nop')
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            sample.list_field += [52]
            sample.save()

        def check_modify_list_iadd(sample_id):
            if False:
                while True:
                    i = 10
            dataset = fo.load_dataset(dataset_name)
            sample = dataset[sample_id]
            self.assertListEqual(sample.list_field, ['fiftyone', 52])
        modify_list_iadd(sample_id)
        check_modify_list_iadd(sample_id)

class MultiProcessSynchronizationTests(unittest.TestCase):
    """Tests that ensure that multiple processes can interact with the database
    simultaneously.
    """
    pass
if __name__ == '__main__':
    fo.config.show_progress_bars = False
    unittest.main(verbosity=2)