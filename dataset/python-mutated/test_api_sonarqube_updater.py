from .dojo_test_case import DojoTestCase
from dojo.models import Tool_Configuration, Tool_Type
from dojo.tools.api_sonarqube.updater import SonarQubeApiUpdater

class TestSonarQubeApiUpdater(DojoTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        tool_type = Tool_Type.objects.create(name='SonarQube')
        Tool_Configuration.objects.create(name='SonarQube', tool_type=tool_type, authentication_type='API')
        self.updater = SonarQubeApiUpdater()

    def test_transitions_for_sonarqube_from_open_1(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('OPEN', 'CONFIRMED'), ['confirm'])

    def test_transitions_for_sonarqube_from_open_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('OPEN', 'RESOLVED / FIXED'), ['resolve'])

    def test_transitions_for_sonarqube_from_reopened_1(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('REOPENED', 'RESOLVED / FIXED'), ['resolve'])

    def test_transitions_for_sonarqube_from_reopened_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('REOPENED', 'CONFIRMED'), ['confirm'])

    def test_transitions_for_sonarqube_from_resolved_1(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('RESOLVED / FIXED', 'CONFIRMED'), ['reopen', 'confirm'])

    def test_transitions_for_sonarqube_from_resolved_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('RESOLVED / FIXED', 'RESOLVED / FALSE-POSITIVE'), ['reopen', 'falsepositive'])

    def test_transitions_for_sonarqube_from_resolved_3(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('RESOLVED / FIXED', 'RESOLVED / WONTFIX'), ['reopen', 'wontfix'])

    def test_transitions_for_sonarqube_fake_target_origin(self):
        if False:
            return 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('FAKE_STATUS', 'RESOLVED / FIXED'), None)

    def test_transitions_for_sonarqube_fake_target_status(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('RESOLVED / FIXED', 'FAKE_STATUS'), None)

    def test_transitions_for_sonarqube_from_confirmed_1(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('CONFIRMED', 'REOPENED'), ['unconfirm'])

    def test_transitions_for_sonarqube_from_confirmed_2(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('CONFIRMED', 'RESOLVED / FIXED'), ['resolve'])

    def test_transitions_for_open_reopen_status_1(self):
        if False:
            return 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('OPEN', 'REOPENED'), None)

    def test_transitions_for_open_reopen_status_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('REOPENED', 'OPEN'), None)

    def test_transitions_for_open_reopen_status_3(self):
        if False:
            return 10
        self.assertEqual(self.updater.get_sonarqube_required_transitions_for('REOPENED', 'REOPENED'), None)