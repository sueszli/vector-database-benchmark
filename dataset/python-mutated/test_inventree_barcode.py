"""Unit tests for InvenTreeBarcodePlugin."""
from django.urls import reverse
import part.models
import stock.models
from InvenTree.unit_test import InvenTreeAPITestCase

class TestInvenTreeBarcode(InvenTreeAPITestCase):
    """Tests for the integrated InvenTreeBarcode barcode plugin."""
    fixtures = ['category', 'part', 'location', 'stock', 'company', 'supplier_part']

    def test_assign_errors(self):
        if False:
            for i in range(10):
                print('nop')
        'Test error cases for assignment action.'

        def test_assert_error(barcode_data):
            if False:
                return 10
            response = self.post(reverse('api-barcode-link'), format='json', data={'barcode': barcode_data, 'stockitem': 521}, expected_code=400)
            self.assertIn('error', response.data)
        test_assert_error('{"stockitem": 521}')
        test_assert_error('{"stocklocation": 7}')
        test_assert_error('{"part": 10004}')

    def assign(self, data, expected_code=None):
        if False:
            for i in range(10):
                print('nop')
        "Perform a 'barcode assign' request"
        return self.post(reverse('api-barcode-link'), data=data, expected_code=expected_code)

    def unassign(self, data, expected_code=None):
        if False:
            return 10
        "Perform a 'barcode unassign' request"
        return self.post(reverse('api-barcode-unlink'), data=data, expected_code=expected_code)

    def scan(self, data, expected_code=None):
        if False:
            print('Hello World!')
        "Perform a 'scan' operation"
        return self.post(reverse('api-barcode-scan'), data=data, expected_code=expected_code)

    def test_unassign_errors(self):
        if False:
            while True:
                i = 10
        'Test various error conditions for the barcode unassign endpoint'
        response = self.unassign({}, expected_code=400)
        self.assertIn('Missing data: Provide one of', str(response.data['error']))
        response = self.unassign({'stockitem': 'abcde', 'part': 'abcde'}, expected_code=400)
        self.assertIn('Multiple conflicting fields:', str(response.data['error']))
        response = self.unassign({'stockitem': 'invalid'}, expected_code=400)
        self.assertIn('No match found', str(response.data['stockitem']))
        response = self.unassign({'part': 'invalid'}, expected_code=400)
        self.assertIn('No match found', str(response.data['part']))

    def test_assign_to_stock_item(self):
        if False:
            print('Hello World!')
        'Test that we can assign a unique barcode to a StockItem object'
        response = self.assign({'barcode': 'abcde'}, expected_code=400)
        self.assertIn('Missing data:', str(response.data))
        response = self.assign({'barcode': 'abcdefg', 'part': 1, 'stockitem': 1}, expected_code=403)
        self.assignRole('part.change')
        self.assignRole('stock.change')
        response = self.assign({'barcode': 'abcdefg', 'part': 1, 'stockitem': 1}, expected_code=200)
        self.assertIn('Assigned barcode to part instance', str(response.data))
        self.assertEqual(response.data['part']['pk'], 1)
        bc_data = '{"blbla": 10007}'
        response = self.assign(data={'barcode': bc_data, 'stockitem': 521}, expected_code=200)
        data = response.data
        self.assertEqual(data['barcode_data'], bc_data)
        self.assertEqual(data['stockitem']['pk'], 521)
        si = stock.models.StockItem.objects.get(pk=521)
        self.assertEqual(si.barcode_data, bc_data)
        self.assertEqual(si.barcode_hash, '2f5dba5c83a360599ba7665b2a4131c6')
        response = self.assign(data={'barcode': bc_data, 'stockitem': 1}, expected_code=400)
        self.assertIn('Barcode matches existing item', str(response.data))
        response = self.unassign({'stockitem': 521}, expected_code=200)
        si.refresh_from_db()
        self.assertEqual(si.barcode_data, '')
        self.assertEqual(si.barcode_hash, '')

    def test_assign_to_part(self):
        if False:
            while True:
                i = 10
        'Test that we can assign a unique barcode to a Part instance'
        barcode = 'xyz-123'
        self.assignRole('part.change')
        response = self.scan({'barcode': barcode}, expected_code=400)
        self.assignRole('part.change')
        response = self.assign({'barcode': barcode, 'part': 99999999}, expected_code=400)
        self.assertIn('No matching part instance found in database', str(response.data))
        response = self.assign({'barcode': barcode, 'part': 1}, expected_code=200)
        self.assertEqual(response.data['part']['pk'], 1)
        self.assertEqual(response.data['success'], 'Assigned barcode to part instance')
        p = part.models.Part.objects.get(pk=1)
        self.assertEqual(p.barcode_data, 'xyz-123')
        self.assertEqual(p.barcode_hash, 'bc39d07e9a395c7b5658c231bf910fae')
        response = self.scan({'barcode': barcode}, expected_code=200)
        self.assertIn('success', response.data)
        self.assertEqual(response.data['plugin'], 'InvenTreeBarcode')
        self.assertEqual(response.data['part']['pk'], 1)
        response = self.assign({'barcode': barcode, 'part': 2}, expected_code=400)
        self.assertIn('Barcode matches existing item', str(response.data['error']))
        self.assignRole('part.change')
        response = self.unassign({'part': 1}, expected_code=200)
        p.refresh_from_db()
        self.assertEqual(p.barcode_data, '')
        self.assertEqual(p.barcode_hash, '')

    def test_assign_to_location(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we can assign a unique barcode to a StockLocation instance'
        barcode = '555555555555555555555555'
        response = self.assign(data={'barcode': barcode, 'stocklocation': 1}, expected_code=403)
        self.assignRole('stock_location.change')
        response = self.assign(data={'barcode': barcode, 'stocklocation': 1}, expected_code=200)
        self.assertIn('success', response.data)
        self.assertEqual(response.data['stocklocation']['pk'], 1)
        loc = stock.models.StockLocation.objects.get(pk=1)
        self.assertEqual(loc.barcode_data, barcode)
        self.assertEqual(loc.barcode_hash, '4aa63f5e55e85c1f842796bf74896dbb')
        response = self.assign(data={'barcode': barcode, 'stocklocation': 2}, expected_code=400)
        self.assertIn('Barcode matches existing item', str(response.data['error']))
        response = self.unassign({'stocklocation': 1}, expected_code=200)
        loc.refresh_from_db()
        self.assertEqual(loc.barcode_data, '')
        self.assertEqual(loc.barcode_hash, '')

    def test_scan_third_party(self):
        if False:
            while True:
                i = 10
        'Test scanning of third-party barcodes'
        response = self.scan({'barcode': 'blbla=10008'}, expected_code=400)
        self.assertEqual(response.data['error'], 'No match found for barcode data')
        response = self.scan({'barcode': 'blbla=10004'}, expected_code=200)
        self.assertEqual(response.data['barcode_data'], 'blbla=10004')
        self.assertEqual(response.data['plugin'], 'InvenTreeBarcode')
        si = stock.models.StockItem.objects.get(pk=1)
        for barcode in ['abcde', 'ABCDE', '12345']:
            si.assign_barcode(barcode_data=barcode)
            response = self.scan({'barcode': barcode}, expected_code=200)
            self.assertIn('success', response.data)
            self.assertEqual(response.data['stockitem']['pk'], 1)

    def test_scan_inventree(self):
        if False:
            i = 10
            return i + 15
        'Test scanning of first-party barcodes'
        response = self.scan({'barcode': '{"stockitem": 5}'}, expected_code=400)
        self.assertIn('No match found for barcode data', str(response.data))
        response = self.scan({'barcode': '{"stockitem": 1}'}, expected_code=200)
        self.assertIn('success', response.data)
        self.assertIn('stockitem', response.data)
        self.assertEqual(response.data['stockitem']['pk'], 1)
        response = self.scan({'barcode': '{"stocklocation": 5}'}, expected_code=200)
        self.assertIn('success', response.data)
        self.assertEqual(response.data['stocklocation']['pk'], 5)
        self.assertEqual(response.data['stocklocation']['api_url'], '/api/stock/location/5/')
        self.assertEqual(response.data['stocklocation']['web_url'], '/stock/location/5/')
        self.assertEqual(response.data['plugin'], 'InvenTreeBarcode')
        response = self.scan({'barcode': '{"part": 5}'}, expected_code=200)
        self.assertEqual(response.data['part']['pk'], 5)
        response = self.scan({'barcode': '{"supplierpart": 1}'}, expected_code=200)
        self.assertEqual(response.data['supplierpart']['pk'], 1)
        self.assertEqual(response.data['plugin'], 'InvenTreeBarcode')
        self.assertIn('success', response.data)
        self.assertIn('barcode_data', response.data)
        self.assertIn('barcode_hash', response.data)