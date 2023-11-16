from google.cloud import compute_v1

def get_image(project_id: str, image_name: str) -> compute_v1.Image:
    if False:
        return 10
    '\n    Retrieve detailed information about a single image from a project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to list images from.\n        image_name: name of the image you want to get details of.\n\n    Returns:\n        An instance of compute_v1.Image object with information about specified image.\n    '
    image_client = compute_v1.ImagesClient()
    return image_client.get(project=project_id, image=image_name)