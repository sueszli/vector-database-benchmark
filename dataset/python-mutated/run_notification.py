def run_notification(transfer_config_name, pubsub_topic):
    if False:
        print('Hello World!')
    orig_transfer_config_name = transfer_config_name
    orig_pubsub_topic = pubsub_topic
    transfer_config_name = 'projects/1234/locations/us/transferConfigs/abcd'
    pubsub_topic = 'projects/PROJECT-ID/topics/TOPIC-ID'
    transfer_config_name = orig_transfer_config_name
    pubsub_topic = orig_pubsub_topic
    from google.cloud import bigquery_datatransfer
    from google.protobuf import field_mask_pb2
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    transfer_config = bigquery_datatransfer.TransferConfig(name=transfer_config_name)
    transfer_config.notification_pubsub_topic = pubsub_topic
    update_mask = field_mask_pb2.FieldMask(paths=['notification_pubsub_topic'])
    transfer_config = transfer_client.update_transfer_config({'transfer_config': transfer_config, 'update_mask': update_mask})
    print(f"Updated config: '{transfer_config.name}'")
    print(f"Notification Pub/Sub topic: '{transfer_config.notification_pubsub_topic}'")
    return transfer_config