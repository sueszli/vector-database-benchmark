from .dojo_test_case import DojoTestCase
from dojo.models import System_Settings

class TestSystemSettings(DojoTestCase):

    def test_system_settings_update(self):
        if False:
            return 10
        try:
            system_settings = System_Settings.objects.get()
        except System_Settings.DoesNotExist:
            system_settings = System_Settings()
        system_settings.enable_jira = True
        system_settings.save()
        system_settings = System_Settings.objects.get()
        self.assertEqual(system_settings.enable_jira, True)
        system_settings.enable_jira = False
        system_settings.save()
        system_settings = System_Settings.objects.get()
        self.assertEqual(system_settings.enable_jira, False)
        system_settings.enable_jira = True
        system_settings.save()
        system_settings = System_Settings.objects.get(no_cache=True)
        self.assertEqual(system_settings.enable_jira, True)