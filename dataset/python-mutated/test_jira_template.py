from .dojo_test_case import DojoTestCase
from dojo.models import Product
from dojo.jira_link import helper as jira_helper
import logging
logger = logging.getLogger(__name__)

class JIRATemplatetTest(DojoTestCase):
    fixtures = ['dojo_testdata.json']

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        DojoTestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        if False:
            while True:
                i = 10
        self.system_settings(enable_jira=True)

    def test_get_jira_issue_template_dir_from_project(self):
        if False:
            i = 10
            return i + 15
        product = Product.objects.get(id=1)
        jira_project = jira_helper.get_jira_project(product)
        jira_project.issue_template_dir = 'issue-trackers/jira_full_extra'
        jira_project.save()
        self.assertEqual(jira_helper.get_jira_issue_template(product), 'issue-trackers/jira_full_extra/jira-description.tpl')

    def test_get_jira_issue_template_dir_from_instance(self):
        if False:
            for i in range(10):
                print('nop')
        product = Product.objects.get(id=1)
        jira_project = jira_helper.get_jira_project(product)
        jira_project.issue_template_dir = None
        jira_project.save()
        self.assertEqual(jira_helper.get_jira_issue_template(product), 'issue-trackers/jira_full/jira-description.tpl')

    def test_get_jira_project_and_instance_no_issue_template_dir(self):
        if False:
            return 10
        product = Product.objects.get(id=1)
        jira_project = jira_helper.get_jira_project(product)
        jira_project.issue_template_dir = None
        jira_project.save()
        jira_instance = jira_helper.get_jira_instance(product)
        jira_instance.issue_template_dir = None
        jira_instance.save()
        self.assertEqual(jira_helper.get_jira_issue_template(product), 'issue-trackers/jira_full/jira-description.tpl')