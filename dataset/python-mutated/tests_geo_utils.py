import frappe
from frappe.geo.utils import get_coords
from frappe.tests.utils import FrappeTestCase

class TestGeoUtils(FrappeTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.todo = frappe.get_doc(dict(doctype='ToDo', description='Test description', assigned_by='Administrator')).insert()
        self.test_location_dict = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Point', 'coordinates': [49.20433, 55.753395]}}]}
        self.test_location = frappe.get_doc({'name': 'Test Location', 'doctype': 'Location', 'location': str(self.test_location_dict)})
        self.test_filter_exists = [['Location', 'name', 'like', '%Test Location%']]
        self.test_filter_not_exists = [['Location', 'name', 'like', '%Test Location Not exists%']]
        self.test_filter_todo = [['ToDo', 'description', 'like', '%Test description%']]

    def test_get_coords_location_with_filter_exists(self):
        if False:
            while True:
                i = 10
        coords = get_coords('Location', self.test_filter_exists, 'location_field')
        self.assertEqual(self.test_location_dict['features'][0]['geometry'], coords['features'][0]['geometry'])

    def test_get_coords_location_with_filter_not_exists(self):
        if False:
            for i in range(10):
                print('nop')
        coords = get_coords('Location', self.test_filter_not_exists, 'location_field')
        self.assertEqual(coords, {'type': 'FeatureCollection', 'features': []})

    def test_get_coords_from_not_existable_location(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(frappe.ValidationError, get_coords, 'ToDo', self.test_filter_todo, 'location_field')

    def test_get_coords_from_not_existable_coords(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(frappe.ValidationError, get_coords, 'ToDo', self.test_filter_todo, 'coordinates')

    def tearDown(self):
        if False:
            print('Hello World!')
        self.todo.delete()