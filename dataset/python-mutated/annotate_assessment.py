from google.cloud import recaptchaenterprise_v1

def annotate_assessment(project_id: str, assessment_id: str) -> None:
    if False:
        print('Hello World!')
    "Pre-requisite: Create an assessment before annotating.\n        Annotate an assessment to provide feedback on the correctness of recaptcha prediction.\n    Args:\n        project_id: Google Cloud Project ID\n        assessment_id: Value of the 'name' field returned from the create_assessment() call.\n    "
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    assessment_name = f'projects/{project_id}/assessments/{assessment_id}'
    request = recaptchaenterprise_v1.AnnotateAssessmentRequest()
    request.name = assessment_name
    request.annotation = request.Annotation.FRAUDULENT
    request.reasons = [request.Reason.FAILED_TWO_FACTOR]
    client.annotate_assessment(request)
    print('Annotated response sent successfully ! ')