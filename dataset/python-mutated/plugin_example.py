from core.models import Image
from django_images.models import Thumbnail

class Plugin:

    def process_image_pre_creation(self, django_settings, image_instance: Image):
        if False:
            while True:
                i = 10
        pass

    def process_thumbnail_pre_creation(self, django_settings, thumbnail_instance: Thumbnail):
        if False:
            for i in range(10):
                print('nop')
        pass