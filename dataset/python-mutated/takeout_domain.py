"""Domain objects for Takeout."""
from __future__ import annotations
from typing import Dict, List

class TakeoutData:
    """Domain object for all information exported as part of Takeout."""

    def __init__(self, user_data: Dict[str, Dict[str, str]], user_images: List[TakeoutImage]) -> None:
        if False:
            while True:
                i = 10
        "Constructs a TakeoutData domain object.\n\n        Args:\n            user_data: dict. The user's Takeout data stored as a dictionary. The\n                dictionary is constructed via takeout_service.py, and the format\n                of the dictionary's contents can be found there.\n            user_images: list(TakeoutImage). A list of TakeoutImage objects\n                representing the user's images.\n        "
        self.user_data = user_data
        self.user_images = user_images

class TakeoutImage:
    """Domain object for storing Base64 image data and the Takeout export path
    for a single image.
    """

    def __init__(self, b64_image_data: str, image_export_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Constructs a TakeoutImage domain object.\n\n        Args:\n            b64_image_data: str. A Base64-encoded string representing the image.\n            image_export_path: str. The path within the images/ folder to write\n                image to in the final Takeout zip.\n        '
        self.b64_image_data = b64_image_data
        self.image_export_path = image_export_path