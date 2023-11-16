import logging
from time import time
from .camera import Camera
from .image import Image
from .scene import Scene
from .randommini import Random
MODEL_FORMAT_ID = '#MiniLight'
logger = logging.getLogger(__name__)

def make_perf_test(filename):
    if False:
        while True:
            i = 10
    '\n    Single core CPU performance test.\n\n    ----------------------------------------------------------------------\n      MiniLight 1.6 Python\n\n      Harrison Ainsworth / HXA7241 and Juraj Sukop : 2007-2008, 2013.\n      http://www.hxa.name/minilight\n\n      2013-05-04\n    ----------------------------------------------------------------------\n\n    MiniLight is a minimal global illumination renderer.\n\n    The model text file format is:\n      #MiniLight\n\n      iterations\n\n      imagewidth imageheight\n      viewposition viewdirection viewangle\n\n      skyemission groundreflection\n\n      vertex0 vertex1 vertex2 reflectivity emitivity\n      vertex0 vertex1 vertex2 reflectivity emitivity\n      ...\n\n    - where iterations and image values are integers, viewangle is a real,\n    and all other values are three parenthised reals. The file must end\n    with a newline. E.g.:\n      #MiniLight\n\n      100\n\n      200 150\n      (0 0.75 -2) (0 0 1) 45\n\n      (3626 5572 5802) (0.1 0.09 0.07)\n\n      (0 0 0) (0 1 0) (1 1 0)  (0.7 0.7 0.7) (0 0 0)\n    '
    model_file_pathname = filename
    model_file = open(model_file_pathname, 'r')
    if model_file.readline().strip() != MODEL_FORMAT_ID:
        raise Exception('invalid model file')
    for line in model_file:
        if not line.isspace():
            iterations = int(line)
            break
    image = Image(model_file)
    camera = Camera(model_file)
    scene = Scene(model_file, camera.view_position)
    model_file.close()
    duration: float = render_taskable(image, camera, scene, iterations)
    num_samples = image.width * image.height * iterations
    logger.debug('Summary: Rendering scene with %d rays took %d seconds giving an average speed of %f rays/s', num_samples, duration, float(num_samples) / duration)
    average = float(num_samples) / duration
    return average

def timedafunc(function):
    if False:
        for i in range(10):
            print('nop')

    def timedExecution(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        t0 = time()
        function(*args, **kwargs)
        t1 = time()
        return t1 - t0
    return timedExecution

@timedafunc
def render_taskable(image, camera, scene, num_samples):
    if False:
        print('Hello World!')
    random = Random()
    aspect = float(image.height) / float(image.width)
    for y in range(image.height):
        for x in range(image.width):
            r = camera.pixel_accumulated_radiance(scene, random, image.width, image.height, x, y, aspect, num_samples)
            image.add_to_pixel(x, y, r)