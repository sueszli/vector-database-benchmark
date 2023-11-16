from google.cloud import contact_center_insights_v1

def create_issue_model(project_id: str) -> contact_center_insights_v1.IssueModel:
    if False:
        return 10
    "Creates an issue model.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n\n    Returns:\n        An issue model.\n    "
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    issue_model = contact_center_insights_v1.IssueModel()
    issue_model.display_name = 'my-model'
    issue_model.input_data_config.filter = 'medium="CHAT"'
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    issue_model_operation = insights_client.create_issue_model(parent=parent, issue_model=issue_model)
    issue_model = issue_model_operation.result(timeout=86400)
    print(f'Created an issue model named {issue_model.name}')
    return issue_model