"""Demo for receiving notifications."""

def receive_notifications(project_id, subscription_name):
    if False:
        while True:
            i = 10
    import concurrent
    from google.cloud import pubsub_v1
    from google.cloud.securitycenter_v1 import NotificationMessage

    def callback(message):
        if False:
            return 10
        print(f'Received message: {message.data}')
        notification_msg = NotificationMessage.from_json(message.data)
        print('Notification config name: {}'.format(notification_msg.notification_config_name))
        print(f'Finding: {notification_msg.finding}')
        message.ack()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_name)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print(f'Listening for messages on {subscription_path}...\n')
    try:
        streaming_pull_future.result(timeout=1)
    except concurrent.futures.TimeoutError:
        streaming_pull_future.cancel()
    return True