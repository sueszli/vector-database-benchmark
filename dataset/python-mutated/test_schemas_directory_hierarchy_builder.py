from unittest import TestCase
from samcli.lib.schemas.schemas_directory_hierarchy_builder import get_package_hierarchy, sanitize_name

class TestSchemasDirectoryHierarchyBuilder(TestCase):

    def test_get_package_hierarchy_for_partner_schema(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('schema.aws.partner.mongodb_com_1234567_tickets.ticketcreated', get_package_hierarchy('aws.partner-mongodb.com/1234567-tickets@TicketCreated'))
        self.assertEqual('schema.aws.partner.zendesk_com_some_other_special_chars.myevent', get_package_hierarchy('aws.partner-zendesk.com/some#other#special#chars@MyEvent'))
        self.assertEqual('schema.aws.partner.pagerduty_com.yougotpaged', get_package_hierarchy('aws.partner-pagerduty.com@YouGotPaged'))

    def test_get_package_hierarchy_for_aws_schema(self):
        if False:
            return 10
        self.assertEqual('schema.aws.autoscalling.ec2instancelaunchsuccessful', get_package_hierarchy('aws.autoscalling@EC2InstanceLaunchSuccessful'))
        self.assertEqual('schema.aws.ec2.ec2instancestatechangenotificationevent', get_package_hierarchy('aws.ec2.EC2InstanceStateChangeNotificationEvent'))

    def test_get_package_hierarchy_for_discovered_event(self):
        if False:
            return 10
        self.assertEqual('schema.order.neworder', get_package_hierarchy('order@NewOrder'))

    def test_get_package_hierarchy_for_customer_uploaded_event(self):
        if False:
            while True:
                i = 10
        self.assertEqual('schema.myevent', get_package_hierarchy('MyEvent'))
        self.assertEqual('schema.myevent_special_characters_etc', get_package_hierarchy('MyEvent.Special#Characters$etc'))
        self.assertEqual('schema.myevent.discriminator', get_package_hierarchy('MyEvent@Discriminator'))
        self.assertEqual('schema.myevent.discriminator.another', get_package_hierarchy('MyEvent@Discriminator@Another'))

    def test_sanitize_root_schema_name_with_dot(self):
        if False:
            i = 10
            return i + 15
        root_schema_name = sanitize_name('MongoDBDatabaseTriggerForMy_store.reviews')
        self.assertEqual(root_schema_name, 'MongoDBDatabaseTriggerForMy_store_reviews')

    def test_sanitize_root_schema_name(self):
        if False:
            for i in range(10):
                print('nop')
        root_schema_name = sanitize_name('MongoDBDatabaseTriggerForMy-store.reviews@hello')
        self.assertEqual(root_schema_name, 'MongoDBDatabaseTriggerForMy_store_reviews.hello')