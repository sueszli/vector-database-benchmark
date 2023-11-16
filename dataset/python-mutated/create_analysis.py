from google.cloud import contact_center_insights_v1

def create_analysis(conversation_name: str) -> contact_center_insights_v1.Analysis:
    if False:
        while True:
            i = 10
    "Creates an analysis.\n\n    Args:\n        conversation_name:\n            The parent resource of the analysis.\n            Format is 'projects/{project_id}/locations/{location_id}/conversations/{conversation_id}'.\n            For example, 'projects/my-project/locations/us-central1/conversations/123456789'.\n\n    Returns:\n        An analysis.\n    "
    analysis = contact_center_insights_v1.Analysis()
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    analysis_operation = insights_client.create_analysis(parent=conversation_name, analysis=analysis)
    analysis = analysis_operation.result(timeout=86400)
    print(f'Created {analysis.name}')
    return analysis