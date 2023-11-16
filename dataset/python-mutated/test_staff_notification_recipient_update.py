import graphene
from ....tests.utils import get_graphql_content
MUTATION_STAFF_NOTIFICATION_RECIPIENT_UPDATE = '\n    mutation StaffNotificationRecipient (\n        $id: ID!,\n        $input: StaffNotificationRecipientInput!\n    ) {\n        staffNotificationRecipientUpdate(id: $id, input: $input) {\n            staffNotificationRecipient {\n                active\n                email\n                user {\n                    firstName\n                    lastName\n                    email\n                }\n            }\n            errors {\n                field\n                message\n                code\n            }\n        }\n    }\n'

def test_staff_notification_update_mutation(staff_api_client, staff_user, permission_manage_settings, staff_notification_recipient):
    if False:
        for i in range(10):
            print('nop')
    old_email = staff_notification_recipient.get_email()
    assert staff_notification_recipient.active
    staff_notification_recipient_id = graphene.Node.to_global_id('StaffNotificationRecipient', staff_notification_recipient.id)
    variables = {'id': staff_notification_recipient_id, 'input': {'active': False}}
    staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_UPDATE, variables, permissions=[permission_manage_settings])
    staff_notification_recipient.refresh_from_db()
    assert not staff_notification_recipient.active
    assert staff_notification_recipient.get_email() == old_email

def test_staff_notification_update_mutation_with_empty_user(staff_api_client, staff_user, permission_manage_settings, staff_notification_recipient):
    if False:
        while True:
            i = 10
    staff_notification_recipient_id = graphene.Node.to_global_id('StaffNotificationRecipient', staff_notification_recipient.id)
    variables = {'id': staff_notification_recipient_id, 'input': {'user': ''}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_UPDATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    staff_notification_recipient.refresh_from_db()
    assert content['data']['staffNotificationRecipientUpdate'] == {'staffNotificationRecipient': None, 'errors': [{'code': 'INVALID', 'field': 'staffNotification', 'message': 'User and email cannot be set empty'}]}

def test_staff_notification_update_mutation_with_empty_email(staff_api_client, staff_user, permission_manage_settings, staff_notification_recipient):
    if False:
        return 10
    staff_notification_recipient_id = graphene.Node.to_global_id('StaffNotificationRecipient', staff_notification_recipient.id)
    variables = {'id': staff_notification_recipient_id, 'input': {'email': ''}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_UPDATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    staff_notification_recipient.refresh_from_db()
    assert content['data']['staffNotificationRecipientUpdate'] == {'staffNotificationRecipient': None, 'errors': [{'code': 'INVALID', 'field': 'staffNotification', 'message': 'User and email cannot be set empty'}]}