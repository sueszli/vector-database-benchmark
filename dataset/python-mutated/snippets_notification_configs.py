"""Demos for working with notification configs."""

def create_notification_config(parent_id, notification_config_id, pubsub_topic):
    if False:
        print('Hello World!')
    '\n    Args:\n        parent_id: must be in one of the following formats:\n            "organizations/{organization_id}"\n            "projects/{project_id}"\n            "folders/{folder_id}"\n        notification_config_id: "your-config-id"\n        pubsub_topic: "projects/{your-project-id}/topics/{your-topic-ic}"\n\n    Ensure this ServiceAccount has the "pubsub.topics.setIamPolicy" permission on the new topic.\n    '
    from google.cloud import securitycenter as securitycenter
    client = securitycenter.SecurityCenterClient()
    created_notification_config = client.create_notification_config(request={'parent': parent_id, 'config_id': notification_config_id, 'notification_config': {'description': 'Notification for active findings', 'pubsub_topic': pubsub_topic, 'streaming_config': {'filter': 'state = "ACTIVE"'}}})
    print(created_notification_config)
    return created_notification_config

def delete_notification_config(parent_id, notification_config_id):
    if False:
        return 10
    '\n    Args:\n        parent_id: must be in one of the following formats:\n            "organizations/{organization_id}"\n            "projects/{project_id}"\n            "folders/{folder_id}"\n        notification_config_id: "your-config-id"\n    '
    from google.cloud import securitycenter as securitycenter
    client = securitycenter.SecurityCenterClient()
    notification_config_name = f'{parent_id}/notificationConfigs/{notification_config_id}'
    client.delete_notification_config(request={'name': notification_config_name})
    print(f'Deleted notification config: {notification_config_name}')
    return True

def get_notification_config(parent_id, notification_config_id):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        parent_id: must be in one of the following formats:\n            "organizations/{organization_id}"\n            "projects/{project_id}"\n            "folders/{folder_id}"\n        notification_config_id: "your-config-id"\n    '
    from google.cloud import securitycenter as securitycenter
    client = securitycenter.SecurityCenterClient()
    notification_config_name = f'{parent_id}/notificationConfigs/{notification_config_id}'
    notification_config = client.get_notification_config(request={'name': notification_config_name})
    print(f'Got notification config: {notification_config}')
    return notification_config

def list_notification_configs(parent_id):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        parent_id: must be in one of the following formats:\n            "organizations/{organization_id}"\n            "projects/{project_id}"\n            "folders/{folder_id}"\n    '
    from google.cloud import securitycenter as securitycenter
    client = securitycenter.SecurityCenterClient()
    notification_configs_iterator = client.list_notification_configs(request={'parent': parent_id})
    for (i, config) in enumerate(notification_configs_iterator):
        print(f'{i}: notification_config: {config}')
    return notification_configs_iterator

def update_notification_config(parent_id, notification_config_id, pubsub_topic):
    if False:
        return 10
    '\n    Args:\n        parent_id: must be in one of the following formats:\n            "organizations/{organization_id}"\n            "projects/{project_id}"\n            "folders/{folder_id}"\n        notification_config_id: "config-id-to-update"\n        pubsub_topic: "projects/{new-project}/topics/{new-topic}"\n\n    If updating a pubsub_topic, ensure this ServiceAccount has the\n    "pubsub.topics.setIamPolicy" permission on the new topic.\n    '
    from google.cloud import securitycenter as securitycenter
    from google.protobuf import field_mask_pb2
    client = securitycenter.SecurityCenterClient()
    notification_config_name = f'{parent_id}/notificationConfigs/{notification_config_id}'
    updated_description = 'New updated description'
    updated_filter = 'state = "INACTIVE"'
    field_mask = field_mask_pb2.FieldMask(paths=['description', 'pubsub_topic', 'streaming_config.filter'])
    updated_notification_config = client.update_notification_config(request={'notification_config': {'name': notification_config_name, 'description': updated_description, 'pubsub_topic': pubsub_topic, 'streaming_config': {'filter': updated_filter}}, 'update_mask': field_mask})
    print(updated_notification_config)
    return updated_notification_config