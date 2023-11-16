import google.cloud.monitoring_v3 as monitoring_v3

def create_ca_monitor_policy(project_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a monitoring policy that notifies you 30 days before a managed CA expires.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n    '
    alertPolicyServiceClient = monitoring_v3.AlertPolicyServiceClient()
    notificationChannelServiceClient = monitoring_v3.NotificationChannelServiceClient()
    query = "fetch privateca.googleapis.com/CertificateAuthority| metric 'privateca.googleapis.com/ca/cert_chain_expiration'| group_by 5m,[value_cert_chain_expiration_mean: mean(value.cert_chain_expiration)]| every 5m| condition val() < 2.592e+06 's'"
    notification_channel = monitoring_v3.NotificationChannel(type_='email', labels={'email_address': 'python-docs-samples-testing@google.com'})
    channel = notificationChannelServiceClient.create_notification_channel(name=notificationChannelServiceClient.common_project_path(project_id), notification_channel=notification_channel)
    alert_policy = monitoring_v3.AlertPolicy(display_name='policy-name', conditions=[monitoring_v3.AlertPolicy.Condition(display_name='ca-cert-chain-expiration', condition_monitoring_query_language=monitoring_v3.AlertPolicy.Condition.MonitoringQueryLanguageCondition(query=query))], combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.AND, notification_channels=[channel.name])
    policy = alertPolicyServiceClient.create_alert_policy(name=notificationChannelServiceClient.common_project_path(project_id), alert_policy=alert_policy)
    print('Monitoring policy successfully created!', policy.name)