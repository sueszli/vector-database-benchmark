import frappe
from frappe.test_runner import make_test_objects
from frappe.tests.utils import FrappeTestCase
from frappe.utils import format_date, today
from frappe.utils.goal import get_monthly_goal_graph_data, get_monthly_results

class TestGoal(FrappeTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        make_test_objects('Event', reset=True)

    def tearDown(self):
        if False:
            return 10
        frappe.db.delete('Event')

    def test_get_monthly_results(self):
        if False:
            print('Hello World!')
        'Test monthly aggregation values of a field'
        result_dict = get_monthly_results('Event', 'subject', 'creation', filters={'event_type': 'Private'}, aggregation='count')
        self.assertEqual(result_dict.get(format_date(today(), 'MM-yyyy')), 2)

    def test_get_monthly_goal_graph_data(self):
        if False:
            while True:
                i = 10
        'Test for accurate values in graph data (based on test_get_monthly_results)'
        docname = frappe.get_list('Event', filters={'subject': ['=', '_Test Event 1']})[0]['name']
        frappe.db.set_value('Event', docname, 'description', 1)
        data = get_monthly_goal_graph_data('Test', 'Event', docname, 'description', 'description', 'description', 'Event', '', 'description', 'creation', filters={'starts_on': '2014-01-01'}, aggregation='count')
        self.assertEqual(float(data['data']['datasets'][0]['values'][-1]), 1)