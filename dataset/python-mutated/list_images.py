from __future__ import annotations
from collections.abc import Iterable
from google.cloud import compute_v1

def list_images(project_id: str) -> Iterable[compute_v1.Image]:
    if False:
        print('Hello World!')
    '\n    Retrieve a list of images available in given project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to list images from.\n\n    Returns:\n        An iterable collection of compute_v1.Image objects.\n    '
    image_client = compute_v1.ImagesClient()
    return image_client.list(project=project_id)