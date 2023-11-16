import graphene
from ....tests.utils import get_graphql_content
MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE = '\n    mutation StaffNotificationRecipient ($input: StaffNotificationRecipientInput!) {\n        staffNotificationRecipientCreate(input: $input) {\n            staffNotificationRecipient {\n                active\n                email\n                user {\n                    id\n                    firstName\n                    lastName\n                    email\n                }\n            }\n            errors {\n                field\n                message\n                code\n            }\n        }\n    }\n'

def test_staff_notification_create_mutation(staff_api_client, staff_user, permission_manage_settings):
    if False:
        return 10
    user_id = graphene.Node.to_global_id('User', staff_user.id)
    variables = {'input': {'user': user_id}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    assert content['data']['staffNotificationRecipientCreate'] == {'staffNotificationRecipient': {'active': True, 'email': staff_user.email, 'user': {'id': user_id, 'firstName': staff_user.first_name, 'lastName': staff_user.last_name, 'email': staff_user.email}}, 'errors': []}

def test_staff_notification_create_mutation_with_staffs_email(staff_api_client, staff_user, permission_manage_settings):
    if False:
        i = 10
        return i + 15
    user_id = graphene.Node.to_global_id('User', staff_user.id)
    variables = {'input': {'email': staff_user.email}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    assert content['data']['staffNotificationRecipientCreate'] == {'staffNotificationRecipient': {'active': True, 'email': staff_user.email, 'user': {'id': user_id, 'firstName': staff_user.first_name, 'lastName': staff_user.last_name, 'email': staff_user.email}}, 'errors': []}

def test_staff_notification_create_mutation_with_customer_user(staff_api_client, customer_user, permission_manage_settings):
    if False:
        i = 10
        return i + 15
    user_id = graphene.Node.to_global_id('User', customer_user.id)
    variables = {'input': {'user': user_id}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    assert content['data']['staffNotificationRecipientCreate'] == {'staffNotificationRecipient': None, 'errors': [{'code': 'INVALID', 'field': 'user', 'message': 'User has to be staff user'}]}

def test_staff_notification_create_mutation_with_email(staff_api_client, permission_manage_settings, permission_manage_staff):
    if False:
        for i in range(10):
            print('nop')
    staff_email = 'test_email@example.com'
    variables = {'input': {'email': staff_email}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE, variables, permissions=[permission_manage_settings, permission_manage_staff])
    content = get_graphql_content(response)
    assert content['data']['staffNotificationRecipientCreate'] == {'staffNotificationRecipient': {'active': True, 'email': staff_email, 'user': None}, 'errors': []}

def test_staff_notification_create_mutation_with_empty_email(staff_api_client, permission_manage_settings):
    if False:
        for i in range(10):
            print('nop')
    staff_email = ''
    variables = {'input': {'email': staff_email}}
    response = staff_api_client.post_graphql(MUTATION_STAFF_NOTIFICATION_RECIPIENT_CREATE, variables, permissions=[permission_manage_settings])
    content = get_graphql_content(response)
    assert content['data']['staffNotificationRecipientCreate'] == {'staffNotificationRecipient': None, 'errors': [{'code': 'INVALID', 'field': 'staffNotification', 'message': 'User and email cannot be set empty'}]}