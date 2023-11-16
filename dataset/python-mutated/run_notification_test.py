from . import run_notification

def test_run_notification(capsys, transfer_config_name, pubsub_topic):
    if False:
        while True:
            i = 10
    run_notification.run_notification(transfer_config_name=transfer_config_name, pubsub_topic=pubsub_topic)
    (out, _) = capsys.readouterr()
    assert 'Updated config:' in out
    assert transfer_config_name in out
    assert 'Notification Pub/Sub topic:' in out
    assert pubsub_topic in out