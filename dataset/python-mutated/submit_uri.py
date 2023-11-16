from google.cloud import webrisk_v1
from google.cloud.webrisk_v1 import Submission

def submit_uri(project_id: str, uri: str) -> Submission:
    if False:
        print('Hello World!')
    'Submits a URI suspected of containing malicious content to be reviewed.\n\n    Returns a google.longrunning.Operation which, once the review is complete, is updated with its result.\n    You can use the [Pub/Sub API] (https://cloud.google.com/pubsub) to receive notifications for the\n    returned Operation.\n    If the result verifies the existence of malicious content, the site will be added to the\n    Google\'s Social Engineering lists in order to protect users that could get exposed to this\n    threat in the future. Only allow-listed projects can use this method during Early Access.\n\n     Args:\n         project_id: The name of the project that is making the submission.\n         uri: The URI that is being reported for malicious content to be analyzed.\n             uri = "http://testsafebrowsing.appspot.com/s/malware.html"\n\n    Returns:\n        Submission response that contains the URI submitted.\n    '
    webrisk_client = webrisk_v1.WebRiskServiceClient()
    submission = webrisk_v1.Submission()
    submission.uri = uri
    threat_confidence = webrisk_v1.ThreatInfo.Confidence(level=webrisk_v1.ThreatInfo.Confidence.ConfidenceLevel.MEDIUM)
    threat_justification = webrisk_v1.ThreatInfo.ThreatJustification(labels=[webrisk_v1.ThreatInfo.ThreatJustification.JustificationLabel.AUTOMATED_REPORT], comments=['Testing submission'])
    threat_info = webrisk_v1.ThreatInfo(abuse_type=webrisk_v1.types.ThreatType.SOCIAL_ENGINEERING, threat_confidence=threat_confidence, threat_justification=threat_justification)
    threat_discovery = webrisk_v1.ThreatDiscovery(platform=webrisk_v1.ThreatDiscovery.Platform.MACOS, region_codes=['US'])
    request = webrisk_v1.SubmitUriRequest(parent=f'projects/{project_id}', submission=submission, threat_info=threat_info, threat_discovery=threat_discovery)
    response = webrisk_client.submit_uri(request).result(timeout=30)
    return response