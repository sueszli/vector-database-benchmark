from django.urls import reverse
from .dojo_test_case import DojoTestCase
from dojo.models import JIRA_Issue
import json
import logging
import dojo.jira_link.helper as jira_helper
logger = logging.getLogger(__name__)

class JIRAWebhookTest(DojoTestCase):
    fixtures = ['dojo_testdata.json']
    jira_issue_comment_template_json = {'timestamp': 1605117321425, 'webhookEvent': 'comment_created', 'comment': {'self': 'http://www.testjira.com/rest/api/2/issue/2/comment/456843', 'id': '456843', 'author': {'self': 'http://www.testjira.com/rest/api/2/user?username=valentijn', 'name': 'valentijn', 'avatarUrls': {'48x48': 'http://www.testjira.com/secure/useravatar?ownerId=valentijn&avatarId=11101', '24x24': 'http://www.testjira.com/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101', '16x16': 'http://www.testjira.com/secure/useravatar?size=x small&ownerId=valentijn&avatarId=11101', '32x32': 'http://www.testjira.com/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101'}, 'displayName': 'Valentijn Scholten', 'active': 'true', 'timeZone': 'Europe/Amsterdam'}, 'body': 'test2', 'updateAuthor': {'self': 'http://www.testjira.com/rest/api/2/user?username=valentijn', 'name': 'valentijn', 'avatarUrls': {'48x48': 'http://www.testjira.com/secure/useravatar?ownerId=valentijn&avatarId=11101', '24x24': 'http://www.testjira.com/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101', '16x16': 'http://www.testjira.com/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101', '32x32': 'http://www.testjira.com/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101'}, 'displayName': 'Valentijn Scholten', 'active': 'true', 'timeZone': 'Europe/Amsterdam'}, 'created': '2020-11-11T18:55:21.425+0100', 'updated': '2020-11-11T18:55:21.425+0100'}}
    jira_issue_comment_template_json_with_email = {'timestamp': 1605117321425, 'webhookEvent': 'comment_created', 'comment': {'self': 'http://www.testjira.com/rest/api/2/issue/2/comment/456843', 'id': '456843', 'author': {'self': 'http://www.testjira.com/rest/api/2/user?username=valentijn', 'emailAddress': 'darthvaalor@testme.nl', 'avatarUrls': {'48x48': 'http://www.testjira.com/secure/useravatar?ownerId=valentijn&avatarId=11101', '24x24': 'http://www.testjira.com/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101', '16x16': 'http://www.testjira.com/secure/useravatar?size=x small&ownerId=valentijn&avatarId=11101', '32x32': 'http://www.testjira.com/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101'}, 'displayName': 'Valentijn Scholten', 'active': 'true', 'timeZone': 'Europe/Amsterdam'}, 'body': 'test2', 'updateAuthor': {'self': 'http://www.testjira.com/rest/api/2/user?username=valentijn', 'emailAddress': 'darthvaalor@testme.nl', 'avatarUrls': {'48x48': 'http://www.testjira.com/secure/useravatar?ownerId=valentijn&avatarId=11101', '24x24': 'http://www.testjira.com/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101', '16x16': 'http://www.testjira.com/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101', '32x32': 'http://www.testjira.com/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101'}, 'displayName': 'Valentijn Scholten', 'active': 'true', 'timeZone': 'Europe/Amsterdam'}, 'created': '2020-11-11T18:55:21.425+0100', 'updated': '2020-11-11T18:55:21.425+0100'}}
    jira_issue_update_template_string = '\n{\n   "timestamp":1605117321475,\n   "webhookEvent":"jira:issue_updated",\n   "issue_event_type_name":"issue_commented",\n   "user":{\n      "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n      "name":"valentijn",\n      "emailAddress ":"valentijn.scholten@isaac.nl",\n      "avatarUrls":{\n         "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n         "24x24":"http s://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n         "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall& ownerId=valentijn&avatarId=11101",\n         "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n      },\n      "displayName ":"Valentijn Scholten",\n      "active":"true",\n      "timeZone":"Europe/Amsterdam"\n   },\n   "issue":{\n      "id":"2",\n      "self":"https://jira.onpremise.org/rest/api/2/issue/2 ",\n      "key":"ISEC-277",\n      "fields":{\n         "issuetype":{\n            "self":"https://jira.onpremise.org/rest/api/2/issuetype/3",\n            "id":"3",\n            "description":"A task is some piece o f work that can be assigned to a user. This does not always result in a quotation/estimate, as it is often some task that needs to be performe d in the context of an existing contract. ",\n            "iconUrl":"https://jira.onpremise.org/secure/viewavatar?size=xsmall&avatarId=16681&avatarType=issuetype",\n            "name":"Task",\n            "subtask":false,\n            "avatarId":16681\n         },\n         "project":{\n            "self":"https://jira.onpremise.org/rest/api/2/project/13532",\n            "id":"13532",\n            "key":"ISEC",\n            "name":"ISAAC security",\n            "projectTypeKey":"software",\n            "avatarUrls":{\n               "48x48":"https://jira.onpremise.org/secure/projectavatar?avatarId=14803",\n               "24x24":"https://jira.onpremise.org/secure/projectavatar?size=small&avatarId=14803",\n               "16x16":"https://jira.onpremise.org/secure/projectavatar?size=xsmall&avatarId=14803",\n               "32x32":"https://jira.onpremise.org/secure/projectavatar?size=medium&avatarId=14803"\n            },\n            "projectCategory":{\n               "self":"https://jira.onpremise.org/rest/api/2/projectCategory/10032",\n               "id":"10032",\n               "description":"All internal isaac projects.",\n               "name":"isaac internal"\n            }\n         },\n         "fixVersions":[\n         ],\n         "customfield_11440":"0|y02wb8: ",\n                        "resolution":{\n                            "self":"http://www.testjira.com/rest/api/2/resolution/11",\n                            "id":"11",\n                            "description":"Cancelled by the customer.",\n                            "name":"Cancelled"\n                        },\n         "resolutiondate":null,\n         "workratio":"-1",\n         "lastViewed":"2020-11-11T18:54:32.489+0100",\n         "watches":{\n            "self":"https://jira.onpremise.org/rest/api/2/issue/ISEC-277/watchers",\n            "watchCount":1,\n            "isWatching":"true"\n         },\n         "customfield_10060":[\n            "dojo_user(dojo_user)",\n            "valentijn(valentijn)"\n         ],\n         "customfield_10182":null,\n         "created":"2019-04-04T15:38:21.248+0200",\n         "customfield_12043":null,\n         "customfield_10340":null,\n         "customfield_10341":null,\n         "customfield_12045":null,\n         "customfield_10100":null,\n         "priority":{\n            "self":"https://jira.onpremise.org/rest/api/2/priority/5",\n            "iconUrl":"https://jira.onpremise.org/images/icons/priorities/trivial.svg",\n            "name":"Trivial (Sev5)",\n            "id":"5"\n         },\n         "customfield_10740":null,\n         "labels":[\n            "NPM_Test",\n            "defect-dojo",\n            "security"\n         ],\n         "timeestimate":null,\n         "aggregatetimeoriginalestimate":null,\n         "issuelinks":[\n         ],\n         "assignee":{\n            "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n            "name":"valentijn",\n            "emailAddress":"valentijn.scholten@isaac.nl",\n            "avatarUrls":{\n               "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n               "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n               "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n               "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n            },\n            "displayName":"Valentijn Scholten",\n            "active":"true",\n            "timeZone":"Europe/Amsterdam"\n         },\n         "updated":"2020-11-11T18:54:32.155+0100",\n         "status":{\n            "self":"https://jira.onpremise.org/rest/api/2/status/10022",\n            "description":"Incoming/New issues.",\n            "iconUrl":"https://jira.onpremise.org/isaac_content/icons/isaac_status_new.gif",\n            "name":"Closed",\n            "id":"10022",\n            "statusCategory":{\n               "self":"https://jira.onpremise.org/rest/api/2/statuscategory/2",\n               "id":2,\n               "key":"new",\n               "colorName":"blue-gray",\n               "name":"To Do"\n            }\n         },\n         "components":[\n         ],\n         "customfield_10051":"2020-11-11T18:54:32.155+0100",\n         "timeoriginalestimate":null,\n         "customfield_10052":null,\n         "description":"description",\n         "customfield_10010":null,\n         "timetracking":{\n         },\n         "attachment":[\n         ],\n         "aggregatetimeestimate":null,\n         "summary":"Regular Expression Denial of Service - (braces, <2.3.1)",\n         "creator":{\n            "self":"https://jira.onpremise.org/rest/api/2/user?username=dojo_user",\n            "name":"dojo_user",\n            "key":"dojo_user",\n            "emailAddress":"defectdojo@isaac.nl",\n            "avatarUrls":{\n               "48x48":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=48",\n               "24x24":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=24",\n               "16x16":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=16",\n               "32x32":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=32"\n            },\n            "displayName":"Defect Dojo",\n            "active":"true",\n            "timeZone":"Europe/Amsterdam"\n         },\n         "subtasks":[\n         ],\n         "customfield_10240":"9223372036854775807",\n         "reporter":{\n            "self":"https://jira.onpremise.org/rest/api/2/user?username=dojo_user",\n            "name":"dojo_user",\n            "key":"dojo_user",\n            "emailAddress":"defectdojo@isaac.nl",\n            "avatarUrls":{\n               "48x48":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=48",\n               "24x24":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=24",\n               "16x16":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=16",\n               "32x32":"https://www.gravatar.com/avatar/9637bfb970eff6176357df615f548f1c?d=mm&s=32"\n            },\n            "displayName":"Defect Dojo",\n            "active":"true",\n            "timeZone":"Europe/Amsterdam"\n         },\n         "aggregateprogress":{\n            "progress":0,\n            "total":0\n         },\n         "customfield_10640":"9223372036854775807",\n         "customfield_10641":null,\n         "environment":null,\n         "duedate":null,\n         "progress":{\n            "progress":0,\n            "total":0\n         },\n         "comment":{\n            "comments":[\n               {\n                  "self":"https://jira.onpremise.org/rest/api/2/issue/2/comment/456841",\n                  "id":"456841",\n                  "author":{\n                     "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n                     "name":"valentijn",\n                     "emailAddress":"valentijn.scholten@isaac.nl",\n                     "avatarUrls":{\n                        "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n                        "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n                        "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n                        "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n                     },\n                     "displayName":"Valentijn Scholten",\n                     "active":"true",\n                     "timeZone":"Europe/Amsterdam"\n                  },\n                  "body":"test comment valentijn",\n                  "updateAuthor":{\n                     "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n                     "name":"valentijn",\n                     "emailAddress":"valentijn.scholten@isaac.nl",\n                     "avatarUrls":{\n                        "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n                        "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n                        "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n                        "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n                     },\n                     "displayName":"Valentijn Scholten",\n                     "active":"true",\n                     "timeZone":"Europe/Amsterdam"\n                  },\n                  "created":"2020-11-11T18:54:32.155+0100",\n                  "updated":"2020-11-11T18:54:32.155+0100"\n               },\n               {\n                  "self":"https://jira.onpremise.org/rest/api/2/issue/2/comment/456843",\n                  "id":"456843",\n                  "author":{\n                     "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n                     "name":"valentijn",\n                     "emailAddress":"valentijn.scholten@isaac.nl",\n                     "avatarUrls":{\n                        "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n                        "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n                        "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n                        "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n                     },\n                     "displayName":"Valentijn Scholten",\n                     "active":"true",\n                     "timeZone":"Europe/Amsterdam"\n                  },\n                  "body":"test2",\n                  "updateAuthor":{\n                     "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n                     "name":"valentijn",\n                     "emailAddress":"valentijn.scholten@isaac.nl",\n                     "avatarUrls":{\n                        "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n                        "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n                        "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n                        "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n                     },\n                     "displayName":"Valentijn Scholten",\n                     "active":"true",\n                     "timeZone":"Europe/Amsterdam"\n                  },\n                  "created":"2020-11-11T18:55:21.425+0100",\n                  "updated":"2020-11-11T18:55:21.425+0100"\n               }\n            ],\n            "maxResults":2,\n            "total":2,\n            "startAt":0\n         },\n         "worklog":{\n            "startAt":0,\n            "maxResults":20,\n            "total":0,\n            "worklogs":[\n            ]\n         }\n      }\n   },\n   "comment":{\n      "self":"https://jira.onpremise.org/rest/api/2/issue/2/comment/456843",\n      "id":"456843",\n      "author":{\n         "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n         "name":"valentijn",\n         "emailAddress":"valentijn.scholten@isaac.nl",\n         "avatarUrls":{\n            "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n            "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n            "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n            "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n         },\n         "displayName":"Valentijn Scholten",\n         "active":"true",\n         "timeZone":"Europe/Amsterdam"\n      },\n      "body":"test2",\n      "updateAuthor":{\n         "self":"https://jira.onpremise.org/rest/api/2/user?username=valentijn",\n         "name":"valentijn",\n         "emailAddress":"valentijn.scholten@isaac.nl",\n         "avatarUrls":{\n            "48x48":"https://jira.onpremise.org/secure/useravatar?ownerId=valentijn&avatarId=11101",\n            "24x24":"https://jira.onpremise.org/secure/useravatar?size=small&ownerId=valentijn&avatarId=11101",\n            "16x16":"https://jira.onpremise.org/secure/useravatar?size=xsmall&ownerId=valentijn&avatarId=11101",\n            "32x32":"https://jira.onpremise.org/secure/useravatar?size=medium&ownerId=valentijn&avatarId=11101"\n         },\n         "displayName":"Valentijn Scholten",\n         "active":"true",\n         "timeZone":"Europe/Amsterdam"\n      },\n      "created":"2020-11-11T18:55:21.425+0100",\n      "updated":"2020-11-11T18:55:21.425+0100"\n   }\n}\n'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        DojoTestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        if False:
            return 10
        self.correct_secret = '12345'
        self.incorrect_secret = '1234567890'

    def test_webhook_get(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get(reverse('jira_web_hook'))
        self.assertEqual(405, response.status_code, response.content[:1000])

    def test_webhook_jira_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.system_settings(enable_jira=False)
        response = self.client.post(reverse('jira_web_hook'))
        self.assertEqual(404, response.status_code, response.content[:1000])

    def test_webhook_disabled(self):
        if False:
            i = 10
            return i + 15
        self.system_settings(enable_jira=False, enable_jira_web_hook=False)
        response = self.client.post(reverse('jira_web_hook'))
        self.assertEqual(404, response.status_code, response.content[:1000])

    def test_webhook_invalid_content_type(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=True)
        response = self.client.post(reverse('jira_web_hook'))
        self.assertEqual(400, response.status_code, response.content[:1000])

    def test_webhook_secret_disabled_no_secret(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=True)
        response = self.client.post(reverse('jira_web_hook'))
        self.assertEqual(400, response.status_code, response.content[:1000])

    def test_webhook_secret_disabled_secret(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=True)
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.incorrect_secret,)))
        self.assertEqual(400, response.status_code, response.content[:1000])

    def test_webhook_secret_enabled_no_secret(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        response = self.client.post(reverse('jira_web_hook'))
        self.assertEqual(403, response.status_code, response.content[:1000])

    def test_webhook_secret_enabled_incorrect_secret(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.incorrect_secret,)))
        self.assertEqual(403, response.status_code, response.content[:1000])

    def test_webhook_secret_enabled_correct_secret(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)))
        self.assertEqual(400, response.status_code, response.content[:1000])

    def test_webhook_comment_on_finding(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_before = finding.notes.count()
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), self.jira_issue_comment_template_json, content_type='application/json')
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_after = finding.notes.count()
        self.assertEqual(200, response.status_code, response.content[:1000])
        self.assertEqual(notes_count_after, notes_count_before + 1)

    def test_webhook_comment_on_finding_from_dojo_note(self):
        if False:
            for i in range(10):
                print('nop')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_before = finding.notes.count()
        body = json.loads(json.dumps(self.jira_issue_comment_template_json))
        body['comment']['updateAuthor']['name'] = 'defect.dojo'
        body['comment']['updateAuthor']['displayName'] = 'Defect Dojo'
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_after = finding.notes.count()
        self.assertEqual(200, response.status_code)
        self.assertEqual(notes_count_after, notes_count_before)

    def test_webhook_comment_on_finding_from_dojo_note_with_email(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_before = finding.notes.count()
        jira_instance = jira_helper.get_jira_instance(finding)
        jira_instance.username = 'defect.dojo@testme.com'
        jira_instance.save()
        body = json.loads(json.dumps(self.jira_issue_comment_template_json_with_email))
        body['comment']['updateAuthor']['emailAddress'] = 'defect.dojo@testme.com'
        body['comment']['updateAuthor']['displayName'] = 'Defect Dojo'
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_after = finding.notes.count()
        jira_instance = jira_helper.get_jira_instance(finding)
        jira_instance.username = 'defect.dojo'
        jira_instance.save()
        self.assertEqual(200, response.status_code)
        self.assertEqual(notes_count_after, notes_count_before)

    def test_webhook_comment_on_finding_jira_under_path(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(json.dumps(self.jira_issue_comment_template_json))
        body['comment']['self'] = 'http://www.testjira.com/my_little_happy_path_for_jira/rest/api/2/issue/2/comment/456843'
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_before = finding.notes.count()
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), self.jira_issue_comment_template_json, content_type='application/json')
        jira_issue = JIRA_Issue.objects.get(jira_id=2)
        finding = jira_issue.finding
        notes_count_after = finding.notes.count()
        self.assertEqual(200, response.status_code, response.content[:1000])
        self.assertEqual(notes_count_after, notes_count_before + 1)

    def test_webhook_comment_on_engagement(self):
        if False:
            i = 10
            return i + 15
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(json.dumps(self.jira_issue_comment_template_json))
        body['comment']['self'] = 'http://www.testjira.com/rest/api/2/issue/333/comment/456843'
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        self.assertEqual(b'Comment for engagement ignored', response.content)

    def test_webhook_update_engagement(self):
        if False:
            for i in range(10):
                print('nop')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(self.jira_issue_update_template_string)
        body['issue']['id'] = 333
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(200, response.status_code, response.content[:1000])
        self.assertEqual(b'Update for engagement ignored', response.content)

    def test_webhook_comment_no_finding_no_engagement(self):
        if False:
            for i in range(10):
                print('nop')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(json.dumps(self.jira_issue_comment_template_json))
        body['comment']['self'] = 'http://www.testjira.com/rest/api/2/issue/666/comment/456843'
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(404, response.status_code, response.content[:1000])

    def test_webhook_update_no_finding_no_engagement(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(self.jira_issue_update_template_string)
        body['issue']['id'] = 999
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(404, response.status_code, response.content[:1000])

    def test_webhook_comment_no_jira_issue_at_all(self):
        if False:
            print('Hello World!')
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(json.dumps(self.jira_issue_comment_template_json))
        body['comment']['self'] = 'http://www.testjira.com/rest/api/2/issue/999/comment/456843'
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(404, response.status_code, response.content[:1000])

    def test_webhook_update_no_jira_issue_at_all(self):
        if False:
            return 10
        self.system_settings(enable_jira=True, enable_jira_web_hook=True, disable_jira_webhook_secret=False, jira_webhook_secret=self.correct_secret)
        body = json.loads(self.jira_issue_update_template_string)
        body['issue']['id'] = 666
        response = self.client.post(reverse('jira_web_hook_secret', args=(self.correct_secret,)), body, content_type='application/json')
        self.assertEqual(404, response.status_code, response.content[:1000])