import inspect
from wagtail.images.exceptions import InvalidFilterSpecError
from wagtail.images.rect import Rect, Vector
from wagtail.images.utils import parse_color_string

class Operation:

    def __init__(self, method, *args):
        if False:
            print('Hello World!')
        self.method = method
        self.args = args
        try:
            inspect.getcallargs(self.construct, *args)
        except TypeError as e:
            raise InvalidFilterSpecError(e)
        try:
            self.construct(*args)
        except ValueError as e:
            raise InvalidFilterSpecError(e)

    def construct(self, *args):
        if False:
            print('Hello World!')
        raise NotImplementedError

class ImageTransform:
    """
    Tracks transformations that are performed on an image.

    This allows multiple transforms to be processed in a single operation and also
    accumulates the operations into a single scale/offset which can be used for
    features such as transforming the focal point of the image.
    """

    def __init__(self, size, image_is_svg=False):
        if False:
            for i in range(10):
                print('nop')
        self._check_size(size, allow_floating_point=image_is_svg)
        self.image_is_svg = image_is_svg
        self.size = size
        self.scale = (1.0, 1.0)
        self.offset = (0.0, 0.0)

    def clone(self):
        if False:
            i = 10
            return i + 15
        clone = ImageTransform(self.size, self.image_is_svg)
        clone.scale = self.scale
        clone.offset = self.offset
        return clone

    def resize(self, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change the image size, stretching the transform to make it fit the new size.\n        '
        self._check_size(size, allow_floating_point=self.image_is_svg)
        clone = self.clone()
        clone.scale = (clone.scale[0] * size[0] / self.size[0], clone.scale[1] * size[1] / self.size[1])
        clone.size = size
        return clone

    def crop(self, rect):
        if False:
            while True:
                i = 10
        '\n        Crop the image to the specified rect.\n        '
        self._check_size(tuple(rect.size), allow_floating_point=self.image_is_svg)
        clone = self.clone()
        clone.offset = (clone.offset[0] - rect.left / self.scale[0], clone.offset[1] - rect.top / self.scale[1])
        clone.size = tuple(rect.size)
        return clone

    def transform_vector(self, vector):
        if False:
            print('Hello World!')
        '\n        Transforms the given vector into the coordinate space of the final image.\n\n        Use this to find out where a point on the source image would end up in the\n        final image after cropping/resizing has been performed.\n\n        Returns a new vector.\n        '
        return Vector((vector.x + self.offset[0]) * self.scale[0], (vector.y + self.offset[1]) * self.scale[1])

    def untransform_vector(self, vector):
        if False:
            while True:
                i = 10
        '\n        Transforms the given vector back to the coordinate space of the source image.\n\n        This performs the inverse of `transform_vector`. Use this to find where a point\n        in the final cropped/resized image originated from in the source image.\n\n        Returns a new vector.\n        '
        return Vector(vector.x / self.scale[0] - self.offset[0], vector.y / self.scale[1] - self.offset[1])

    def get_rect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a Rect representing the region of the original image to be cropped.\n        '
        return Rect(-self.offset[0], -self.offset[1], -self.offset[0] + self.size[0] / self.scale[0], -self.offset[1] + self.size[1] / self.scale[1])

    @staticmethod
    def _check_size(size, allow_floating_point=False):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(size, tuple) or len(size) != 2:
            raise TypeError('Image size must be a 2-tuple')
        if not allow_floating_point and (int(size[0]) != size[0] or int(size[1]) != size[1]):
            raise TypeError('Image size must be a 2-tuple of integers')
        if size[0] < 1 or size[1] < 1:
            raise ValueError('Image width and height must both be 1 or greater')

class TransformOperation(Operation):

    def run(self, image, transform):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class FillOperation(TransformOperation):
    vary_fields = ('focal_point_width', 'focal_point_height', 'focal_point_x', 'focal_point_y')

    def construct(self, size, *extra):
        if False:
            while True:
                i = 10
        (width_str, height_str) = size.split('x')
        self.width = int(width_str)
        self.height = int(height_str)
        self.crop_closeness = 0
        for extra_part in extra:
            if extra_part.startswith('c'):
                self.crop_closeness = int(extra_part[1:])
            else:
                raise ValueError('Unrecognised filter spec part: %s' % extra_part)
        self.crop_closeness /= 100
        if self.crop_closeness > 1:
            self.crop_closeness = 1

    def run(self, transform, image):
        if False:
            return 10
        (image_width, image_height) = transform.size
        focal_point = image.get_focal_point()
        crop_aspect_ratio = self.width / self.height
        crop_max_scale = min(image_width, image_height * crop_aspect_ratio)
        crop_max_width = crop_max_scale
        crop_max_height = crop_max_scale / crop_aspect_ratio
        crop_width = crop_max_width
        crop_height = crop_max_height
        if focal_point is not None:
            crop_min_scale = max(focal_point.width, focal_point.height * crop_aspect_ratio)
            crop_min_width = crop_min_scale
            crop_min_height = crop_min_scale / crop_aspect_ratio
            if not crop_min_scale >= crop_max_scale:
                max_crop_closeness = max(1 - (self.width - crop_min_width) / (crop_max_width - crop_min_width), 1 - (self.height - crop_min_height) / (crop_max_height - crop_min_height))
                crop_closeness = min(self.crop_closeness, max_crop_closeness)
                if 1 >= crop_closeness >= 0:
                    crop_width = crop_max_width + (crop_min_width - crop_max_width) * crop_closeness
                    crop_height = crop_max_height + (crop_min_height - crop_max_height) * crop_closeness
        if focal_point is not None:
            (fp_x, fp_y) = focal_point.centroid
        else:
            fp_x = image_width / 2
            fp_y = image_height / 2
        fp_u = fp_x / image_width
        fp_v = fp_y / image_height
        crop_x = fp_x - (fp_u - 0.5) * crop_width
        crop_y = fp_y - (fp_v - 0.5) * crop_height
        rect = Rect.from_point(crop_x, crop_y, crop_width, crop_height)
        if focal_point is not None:
            rect = rect.move_to_cover(focal_point)
        rect = rect.move_to_clamp(Rect(0, 0, image_width, image_height))
        transform = transform.crop(rect.round())
        (aftercrop_width, aftercrop_height) = transform.size
        scale = self.width / aftercrop_width
        if scale < 1.0:
            transform = transform.resize((self.width, self.height))
        return transform

class MinMaxOperation(TransformOperation):

    def construct(self, size):
        if False:
            return 10
        (width_str, height_str) = size.split('x')
        self.width = int(width_str)
        self.height = int(height_str)

    def run(self, transform, image):
        if False:
            while True:
                i = 10
        (image_width, image_height) = transform.size
        horz_scale = self.width / image_width
        vert_scale = self.height / image_height
        if self.method == 'min':
            if image_width <= self.width or image_height <= self.height:
                return transform
            if horz_scale > vert_scale:
                width = self.width
                height = int(image_height * horz_scale)
            else:
                width = int(image_width * vert_scale)
                height = self.height
        elif self.method == 'max':
            if image_width <= self.width and image_height <= self.height:
                return transform
            if horz_scale < vert_scale:
                width = self.width
                height = int(image_height * horz_scale)
            else:
                width = int(image_width * vert_scale)
                height = self.height
        else:
            return transform
        width = width if width > 0 else 1
        height = height if height > 0 else 1
        return transform.resize((width, height))

class WidthHeightOperation(TransformOperation):

    def construct(self, size):
        if False:
            i = 10
            return i + 15
        self.size = int(size)

    def run(self, transform, image):
        if False:
            print('Hello World!')
        (image_width, image_height) = transform.size
        if self.method == 'width':
            if image_width <= self.size:
                return transform
            scale = self.size / image_width
            width = self.size
            height = int(image_height * scale)
        elif self.method == 'height':
            if image_height <= self.size:
                return transform
            scale = self.size / image_height
            width = int(image_width * scale)
            height = self.size
        else:
            return transform
        width = width if width > 0 else 1
        height = height if height > 0 else 1
        return transform.resize((width, height))

class ScaleOperation(TransformOperation):

    def construct(self, percent):
        if False:
            i = 10
            return i + 15
        self.percent = float(percent)

    def run(self, transform, image):
        if False:
            while True:
                i = 10
        (image_width, image_height) = transform.size
        scale = self.percent / 100
        width = int(image_width * scale)
        height = int(image_height * scale)
        width = width if width > 0 else 1
        height = height if height > 0 else 1
        return transform.resize((width, height))

class FilterOperation(Operation):

    def run(self, willow, image, env):
        if False:
            return 10
        raise NotImplementedError

class DoNothingOperation(FilterOperation):

    def construct(self):
        if False:
            while True:
                i = 10
        pass

    def run(self, willow, image, env):
        if False:
            for i in range(10):
                print('nop')
        return willow

class JPEGQualityOperation(FilterOperation):

    def construct(self, quality):
        if False:
            while True:
                i = 10
        self.quality = int(quality)
        if self.quality > 100:
            raise ValueError('JPEG quality must not be higher than 100')

    def run(self, willow, image, env):
        if False:
            print('Hello World!')
        env['jpeg-quality'] = self.quality

class AvifQualityOperation(FilterOperation):

    def construct(self, quality):
        if False:
            i = 10
            return i + 15
        self.quality = int(quality)
        if self.quality > 100:
            raise ValueError('AVIF quality must not be higher than 100')

    def run(self, willow, image, env):
        if False:
            while True:
                i = 10
        env['avif-quality'] = self.quality

class WebPQualityOperation(FilterOperation):

    def construct(self, quality):
        if False:
            return 10
        self.quality = int(quality)
        if self.quality > 100:
            raise ValueError('WebP quality must not be higher than 100')

    def run(self, willow, image, env):
        if False:
            i = 10
            return i + 15
        env['webp-quality'] = self.quality

class FormatOperation(FilterOperation):
    supported_formats = ['jpeg', 'png', 'gif', 'webp', 'avif']

    def construct(self, format, *options):
        if False:
            return 10
        self.format = format
        self.options = options
        if self.format not in self.supported_formats:
            raise ValueError(f"Format must be one of: {', '.join(self.supported_formats)}. Got: {self.format}")

    def run(self, willow, image, env):
        if False:
            while True:
                i = 10
        env['output-format'] = self.format
        env['output-format-options'] = self.options

class BackgroundColorOperation(FilterOperation):

    def construct(self, color_string):
        if False:
            i = 10
            return i + 15
        self.color = parse_color_string(color_string)

    def run(self, willow, image, env):
        if False:
            for i in range(10):
                print('nop')
        return willow.set_background_color_rgb(self.color)