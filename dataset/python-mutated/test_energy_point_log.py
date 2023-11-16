import frappe
from frappe.desk.form.assign_to import add as assign_to
from frappe.desk.page.user_profile.user_profile import get_energy_points_heatmap_data
from frappe.tests.utils import FrappeTestCase
from frappe.utils.testutils import add_custom_field, clear_custom_fields
from .energy_point_log import create_review_points_log
from .energy_point_log import get_energy_points as _get_energy_points
from .energy_point_log import review

class TestEnergyPointLog(FrappeTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        settings = frappe.get_single('Energy Point Settings')
        settings.enabled = 1
        settings.save()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        settings = frappe.get_single('Energy Point Settings')
        settings.enabled = 0
        settings.save()

    def setUp(self):
        if False:
            while True:
                i = 10
        frappe.cache.delete_value('energy_point_rule_map')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('Administrator')
        frappe.db.delete('Energy Point Log')
        frappe.db.delete('Energy Point Rule')
        frappe.cache.delete_value('energy_point_rule_map')

    def test_user_energy_point(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('test@example.com')
        todo_point_rule = create_energy_point_rule_for_todo()
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(points_after_closing_todo, energy_point_of_user + todo_point_rule.points)
        created_todo.save()
        points_after_double_save = get_points('test@example.com')
        self.assertEqual(points_after_double_save, energy_point_of_user + todo_point_rule.points)

    def test_points_based_on_multiplier_field(self):
        if False:
            i = 10
            return i + 15
        frappe.set_user('test@example.com')
        add_custom_field('ToDo', 'multiplier', 'Float')
        multiplier_value = 0.51
        todo_point_rule = create_energy_point_rule_for_todo('multiplier')
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.multiplier = multiplier_value
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(points_after_closing_todo, energy_point_of_user + round(todo_point_rule.points * multiplier_value))
        clear_custom_fields('ToDo')

    def test_points_based_on_max_points(self):
        if False:
            return 10
        frappe.set_user('test@example.com')
        multiplier_value = 15
        max_points = 50
        add_custom_field('ToDo', 'multiplier', 'Float')
        todo_point_rule = create_energy_point_rule_for_todo('multiplier', max_points=max_points)
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.multiplier = multiplier_value
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertNotEqual(points_after_closing_todo, energy_point_of_user + round(todo_point_rule.points * multiplier_value))
        self.assertEqual(points_after_closing_todo, energy_point_of_user + max_points)
        clear_custom_fields('ToDo')

    def test_disabled_energy_points(self):
        if False:
            print('Hello World!')
        settings = frappe.get_single('Energy Point Settings')
        settings.enabled = 0
        settings.save()
        frappe.set_user('test@example.com')
        create_energy_point_rule_for_todo()
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(points_after_closing_todo, energy_point_of_user)
        settings.enabled = 1
        settings.save()

    def test_review(self):
        if False:
            for i in range(10):
                print('nop')
        created_todo = create_a_todo()
        review_points = 20
        create_review_points_log('test2@example.com', review_points)
        frappe.set_user('test2@example.com')
        review_points_before_review = get_points('test2@example.com', 'review_points')
        self.assertEqual(review_points_before_review, review_points)
        appreciation_points = 5
        energy_points_before_review = get_points('test@example.com')
        review(created_todo, appreciation_points, 'test@example.com', 'good job')
        energy_points_after_review = get_points('test@example.com')
        review_points_after_review = get_points('test2@example.com', 'review_points')
        self.assertEqual(energy_points_after_review, energy_points_before_review + appreciation_points)
        self.assertEqual(review_points_after_review, review_points_before_review - appreciation_points)
        criticism_points = 2
        todo = create_a_todo(description='Bad patch')
        energy_points_before_review = energy_points_after_review
        review_points_before_review = review_points_after_review
        review(todo, criticism_points, 'test@example.com', 'You could have done better.', 'Criticism')
        energy_points_after_review = get_points('test@example.com')
        review_points_after_review = get_points('test2@example.com', 'review_points')
        self.assertEqual(energy_points_after_review, energy_points_before_review - criticism_points)
        self.assertEqual(review_points_after_review, review_points_before_review - criticism_points)

    def test_user_energy_point_as_admin(self):
        if False:
            return 10
        frappe.set_user('Administrator')
        create_energy_point_rule_for_todo()
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        points_after_closing_todo = get_points('Administrator')
        self.assertEqual(points_after_closing_todo, 0)

    def test_revert_points_on_cancelled_doc(self):
        if False:
            while True:
                i = 10
        frappe.set_user('test@example.com')
        create_energy_point_rule_for_todo()
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        energy_point_logs = frappe.get_all('Energy Point Log')
        self.assertEqual(len(energy_point_logs), 1)
        frappe.set_user('Administrator')
        created_todo.docstatus = 1
        created_todo.save()
        created_todo.docstatus = 2
        created_todo.save()
        energy_point_logs = frappe.get_all('Energy Point Log', fields=['reference_name', 'type', 'reverted'])
        self.assertListEqual(energy_point_logs, [{'reference_name': created_todo.name, 'type': 'Revert', 'reverted': 0}, {'reference_name': created_todo.name, 'type': 'Auto', 'reverted': 1}])

    def test_energy_point_for_new_document_creation(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('test@example.com')
        todo_point_rule = create_energy_point_rule_for_todo(for_doc_event='New')
        points_before_todo_creation = get_points('test@example.com')
        create_a_todo()
        points_after_todo_creation = get_points('test@example.com')
        self.assertEqual(points_after_todo_creation, points_before_todo_creation + todo_point_rule.points)

    def test_point_allocation_for_assigned_users(self):
        if False:
            print('Hello World!')
        todo = create_a_todo()
        assign_users_to_todo(todo.name, ['test@example.com', 'test2@example.com'])
        test_user_before_points = get_points('test@example.com')
        test2_user_before_points = get_points('test2@example.com')
        rule = create_energy_point_rule_for_todo(for_assigned_users=1)
        todo.status = 'Closed'
        todo.save()
        test_user_after_points = get_points('test@example.com')
        test2_user_after_points = get_points('test2@example.com')
        self.assertEqual(test_user_after_points, test_user_before_points + rule.points)
        self.assertEqual(test2_user_after_points, test2_user_before_points + rule.points)

    def test_eps_heatmap_query(self):
        if False:
            return 10
        self.assertIsInstance(get_energy_points_heatmap_data(user='test@example.com', date=None), dict)

    def test_points_on_field_value_change(self):
        if False:
            print('Hello World!')
        rule = create_energy_point_rule_for_todo(for_doc_event='Value Change', field_to_check='description')
        frappe.set_user('test@example.com')
        points_before_todo_creation = get_points('test@example.com')
        todo = create_a_todo()
        todo.status = 'Closed'
        todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(points_after_closing_todo, points_before_todo_creation)
        todo.description = 'This is new todo'
        todo.save()
        points_after_changing_todo_description = get_points('test@example.com')
        self.assertEqual(points_after_changing_todo_description, points_before_todo_creation + rule.points)

    def test_apply_only_once(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('test@example.com')
        todo_point_rule = create_energy_point_rule_for_todo(apply_once=True, user_field='modified_by')
        first_user_points = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        first_user_points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(first_user_points_after_closing_todo, first_user_points + todo_point_rule.points)
        frappe.set_user('test2@example.com')
        second_user_points = get_points('test2@example.com')
        created_todo.save(ignore_permissions=True)
        second_user_points_after_closing_todo = get_points('test2@example.com')
        self.assertEqual(second_user_points_after_closing_todo, second_user_points)

    def test_allow_creation_of_new_log_if_the_previous_log_was_reverted(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('test@example.com')
        todo_point_rule = create_energy_point_rule_for_todo()
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        log_name = frappe.db.exists('Energy Point Log', {'reference_name': created_todo.name})
        frappe.get_doc('Energy Point Log', log_name).revert('Just for test')
        points_after_reverting_todo = get_points('test@example.com')
        created_todo.save()
        points_after_saving_todo_again = get_points('test@example.com')
        rule_points = todo_point_rule.points
        self.assertEqual(points_after_closing_todo, energy_point_of_user + rule_points)
        self.assertEqual(points_after_reverting_todo, points_after_closing_todo - rule_points)
        self.assertEqual(points_after_saving_todo_again, points_after_reverting_todo + rule_points)

    def test_energy_points_disabled_user(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.set_user('test@example.com')
        user = frappe.get_doc('User', 'test@example.com')
        user.enabled = 0
        user.save()
        todo_point_rule = create_energy_point_rule_for_todo()
        energy_point_of_user = get_points('test@example.com')
        created_todo = create_a_todo()
        created_todo.status = 'Closed'
        created_todo.save()
        points_after_closing_todo = get_points('test@example.com')
        self.assertEqual(points_after_closing_todo, energy_point_of_user)
        with self.set_user('Administrator'):
            user.enabled = 1
            user.save()
        created_todo.save()
        points_after_re_saving_todo = get_points('test@example.com')
        self.assertEqual(points_after_re_saving_todo, energy_point_of_user + todo_point_rule.points)

def create_energy_point_rule_for_todo(multiplier_field=None, for_doc_event='Custom', max_points=None, for_assigned_users=0, field_to_check=None, apply_once=False, user_field='owner'):
    if False:
        for i in range(10):
            print('nop')
    name = 'ToDo Closed'
    point_rule_exists = frappe.db.exists('Energy Point Rule', name)
    if point_rule_exists:
        return frappe.get_doc('Energy Point Rule', name)
    return frappe.get_doc({'doctype': 'Energy Point Rule', 'rule_name': name, 'points': 5, 'reference_doctype': 'ToDo', 'condition': 'doc.status == "Closed"', 'for_doc_event': for_doc_event, 'user_field': user_field, 'for_assigned_users': for_assigned_users, 'multiplier_field': multiplier_field, 'max_points': max_points, 'field_to_check': field_to_check, 'apply_only_once': apply_once}).insert(ignore_permissions=1)

def create_a_todo(description=None):
    if False:
        while True:
            i = 10
    if not description:
        description = 'Fix a bug'
    return frappe.get_doc({'doctype': 'ToDo', 'description': description}).insert(ignore_permissions=True)

def get_points(user, point_type='energy_points'):
    if False:
        i = 10
        return i + 15
    return _get_energy_points(user).get(point_type) or 0

def assign_users_to_todo(todo_name, users):
    if False:
        return 10
    for user in users:
        assign_to({'assign_to': [user], 'doctype': 'ToDo', 'name': todo_name})