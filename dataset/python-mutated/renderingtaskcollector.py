import logging
import math
from typing import Optional
from apps.rendering.resources.imgrepr import OpenCVImgRepr
logger = logging.getLogger('apps.rendering')

class RenderingTaskCollector(object):

    def __init__(self, width=None, height=None):
        if False:
            return 10
        self.accepted_img_files = []
        self.width = width
        self.height = height
        self.channels = 1
        self.dtype = None

    def add_img_file(self, img_file):
        if False:
            i = 10
            return i + 15
        '\n        Add file path to the image with subtask result\n        :param str img_file: path to the file\n        '
        self.accepted_img_files.append(img_file)

    def finalize(self) -> Optional[OpenCVImgRepr]:
        if False:
            print('Hello World!')
        '\n        Connect all collected files and return final image\n        :return OpenCV Image Representation or None\n        '
        if len(self.accepted_img_files) == 0:
            return None
        return self.finalize_img()

    def finalize_img(self):
        if False:
            return 10
        (res_x, res_y) = (0, 0)
        for name in self.accepted_img_files:
            image = OpenCVImgRepr.from_image_file(name)
            (img_y, res_x) = image.img.shape[:2]
            res_y += img_y
            self.dtype = image.img.dtype
            if len(image.img.shape) == 3:
                self.channels = image.img.shape[2]
        self.width = res_x
        self.height = res_y
        final_img = OpenCVImgRepr.empty(self.width, self.height, self.channels, self.dtype)
        offset = 0
        for img_path in self.accepted_img_files:
            image = OpenCVImgRepr.from_image_file(img_path)
            final_img.paste_image(image, 0, offset)
            offset += image.get_height()
        return final_img

    def _paste_image(self, final_img, new_part, num):
        if False:
            i = 10
            return i + 15
        img_offset = OpenCVImgRepr.empty(self.width, self.height)
        offset = int(math.floor(num * float(self.height) / float(len(self.accepted_img_files))))
        img_offset.paste_image(new_part, 0, offset)
        img_offset.add(final_img)
        return img_offset