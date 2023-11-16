from typing import Optional
from tagging.images_hierarchy import ALL_IMAGES
from tagging.manifests import ManifestInterface
from tagging.taggers import TaggerInterface

def get_taggers_and_manifests(short_image_name: Optional[str]) -> tuple[list[TaggerInterface], list[ManifestInterface]]:
    if False:
        while True:
            i = 10
    if short_image_name is None:
        return [[], []]
    image_description = ALL_IMAGES[short_image_name]
    (parent_taggers, parent_manifests) = get_taggers_and_manifests(image_description.parent_image)
    return (parent_taggers + image_description.taggers, parent_manifests + image_description.manifests)