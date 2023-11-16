"""Unit testing for BOM upload / import functionality."""
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
import tablib
from InvenTree.unit_test import InvenTreeAPITestCase
from part.models import Part

class BomUploadTest(InvenTreeAPITestCase):
    """Test BOM file upload API endpoint."""
    roles = ['part.add', 'part.change']

    @classmethod
    def setUpTestData(cls):
        if False:
            for i in range(10):
                print('nop')
        'Create BOM data as part of setup routine'
        super().setUpTestData()
        Part.objects.rebuild()
        cls.part = Part.objects.create(name='Assembly', description='An assembled part', assembly=True, component=False)
        parts = []
        for i in range(10):
            parts.append(Part(name=f'Component {i}', IPN=f'CMP_{i}', description='A subcomponent that can be used in a BOM', component=True, assembly=False, lft=0, rght=0, level=0, tree_id=0))
        Part.objects.bulk_create(parts)

    def post_bom(self, filename, file_data, clear_existing=None, expected_code=None, content_type='text/plain'):
        if False:
            print('Hello World!')
        'Helper function for submitting a BOM file'
        bom_file = SimpleUploadedFile(filename, file_data, content_type=content_type)
        if clear_existing is None:
            clear_existing = False
        response = self.post(reverse('api-bom-import-upload'), data={'data_file': bom_file}, expected_code=expected_code, format='multipart')
        return response

    def test_missing_file(self):
        if False:
            for i in range(10):
                print('nop')
        'POST without a file.'
        response = self.post(reverse('api-bom-import-upload'), data={}, expected_code=400)
        self.assertIn('No file was submitted', str(response.data['data_file']))

    def test_unsupported_file(self):
        if False:
            for i in range(10):
                print('nop')
        'POST with an unsupported file type.'
        response = self.post_bom('sample.txt', b'hello world', expected_code=400)
        self.assertIn('Unsupported file type', str(response.data['data_file']))

    def test_broken_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test upload with broken (corrupted) files.'
        response = self.post_bom('sample.csv', b'', expected_code=400)
        self.assertIn('The submitted file is empty', str(response.data['data_file']))
        response = self.post_bom('test.xls', b'hello world', expected_code=400, content_type='application/xls')
        self.assertIn('Unsupported format, or corrupt file', str(response.data['data_file']))

    def test_missing_rows(self):
        if False:
            print('Hello World!')
        'Test upload of an invalid file (without data rows)'
        dataset = tablib.Dataset()
        dataset.headers = ['apple', 'banana']
        response = self.post_bom('test.csv', bytes(dataset.csv, 'utf8'), content_type='text/csv', expected_code=400)
        self.assertIn('No data rows found in file', str(response.data))
        response = self.post_bom('bom.xlsx', dataset.xlsx, content_type='application/xlsx', expected_code=400)
        self.assertIn('No data rows found in file', str(response.data))

    def test_missing_columns(self):
        if False:
            return 10
        'Upload extracted data, but with missing columns.'
        url = reverse('api-bom-import-extract')
        rows = [['1', 'test'], ['2', 'test']]
        response = self.post(url, {}, expected_code=400)
        self.assertIn('This field is required', str(response.data['rows']))
        self.assertIn('This field is required', str(response.data['columns']))
        response = self.post(url, {'rows': rows, 'columns': ['part', 'reference']}, expected_code=400)
        self.assertIn("Missing required column: 'quantity'", str(response.data))
        response = self.post(url, {'rows': rows, 'columns': ['quantity', 'reference']}, expected_code=400)
        self.assertIn('No part column specified', str(response.data))
        self.post(url, {'rows': rows, 'columns': ['quantity', 'part']}, expected_code=201)

    def test_invalid_data(self):
        if False:
            print('Hello World!')
        'Upload data which contains errors.'
        dataset = tablib.Dataset()
        dataset.headers = ['part_id', 'quantity']
        components = Part.objects.filter(component=True)
        for (idx, cmp) in enumerate(components):
            if idx == 5:
                cmp.component = False
                cmp.save()
            dataset.append([cmp.pk, idx])
        url = reverse('api-bom-import-extract')
        response = self.post(url, {'columns': dataset.headers, 'rows': list(dataset)})
        rows = response.data['rows']
        self.assertEqual(len(rows), len(dataset))
        for (idx, row) in enumerate(rows):
            data = row['data']
            cmp = components[idx]
            data['part'] = cmp.pk
        self.assertEqual(rows[0]['data']['errors']['quantity'], 'Quantity must be greater than zero')
        self.assertEqual(rows[5]['data']['errors']['part'], 'Part is not designated as a component')

    def test_part_guess(self):
        if False:
            return 10
        "Test part 'guessing' when PK values are not supplied."
        dataset = tablib.Dataset()
        dataset.headers = ['part_name', 'quantity']
        components = Part.objects.filter(component=True)
        for component in components:
            dataset.append([component.name, 10])
        url = reverse('api-bom-import-extract')
        response = self.post(url, {'columns': dataset.headers, 'rows': list(dataset)}, expected_code=201)
        rows = response.data['rows']
        self.assertEqual(len(rows), 10)
        for idx in range(10):
            self.assertEqual(rows[idx]['data']['part'], components[idx].pk)
        dataset = tablib.Dataset()
        dataset.headers = ['part_ipn', 'quantity']
        for component in components:
            dataset.append([component.IPN, 10])
        response = self.post(url, {'columns': dataset.headers, 'rows': list(dataset)}, expected_code=201)
        rows = response.data['rows']
        self.assertEqual(len(rows), 10)
        for idx in range(10):
            self.assertEqual(rows[idx]['data']['part'], components[idx].pk)

    def test_levels(self):
        if False:
            i = 10
            return i + 15
        'Test that multi-level BOMs are correctly handled during upload.'
        url = reverse('api-bom-import-extract')
        dataset = tablib.Dataset()
        dataset.headers = ['level', 'part', 'quantity']
        components = Part.objects.filter(component=True)
        for (idx, cmp) in enumerate(components):
            dataset.append([idx % 3, cmp.pk, 2])
        response = self.post(url, {'rows': list(dataset), 'columns': dataset.headers}, expected_code=201)
        rows = response.data['rows']
        self.assertEqual(len(response.data['rows']), 3)
        self.assertEqual(rows[0]['data']['part'], components[1].pk)
        self.assertEqual(rows[1]['data']['part'], components[4].pk)
        self.assertEqual(rows[2]['data']['part'], components[7].pk)