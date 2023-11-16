from google.cloud import compute_v1

def get_image_from_family(project: str, family: str) -> compute_v1.Image:
    if False:
        i = 10
        return i + 15
    '\n    Retrieve the newest image that is part of a given family in a project.\n\n    Args:\n        project: project ID or project number of the Cloud project you want to get image from.\n        family: name of the image family you want to get image from.\n\n    Returns:\n        An Image object.\n    '
    image_client = compute_v1.ImagesClient()
    newest_image = image_client.get_from_family(project=project, family=family)
    return newest_image