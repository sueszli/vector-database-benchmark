from google.cloud import compute_v1

def set_deprecation_status(project_id: str, image_name: str, status: compute_v1.DeprecationStatus.State) -> None:
    if False:
        print('Hello World!')
    "\n    Modify the deprecation status of an image.\n\n    Note: Image objects by default don't have the `deprecated` attribute at all unless it's set.\n\n    Args:\n        project_id: project ID or project number of the Cloud project that hosts the image.\n        image_name: name of the image you want to modify\n        status: the status you want to set for the image. Available values are available in\n            `compute_v1.DeprecationStatus.State` enum. Learn more about image deprecation statuses:\n            https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images#deprecation-states\n    "
    image_client = compute_v1.ImagesClient()
    deprecation_status = compute_v1.DeprecationStatus()
    deprecation_status.state = status.name
    operation = image_client.deprecate(project=project_id, image=image_name, deprecation_status_resource=deprecation_status)
    wait_for_extended_operation(operation, 'changing deprecation state of an image')