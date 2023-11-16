from io import StringIO
from .camera import Camera
from .scene import Scene

class RenderTaskDesc:

    @classmethod
    def createRenderTaskDesc(cls, id, x, y, w, h, num_pixels, num_samples):
        if False:
            while True:
                i = 10
        return RenderTaskDesc(id, x, y, w, h, num_pixels, num_samples)

    def __init__(self, id, x, y, w, h, num_pixels, num_samples):
        if False:
            while True:
                i = 10
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.num_pixels = num_pixels
        self.num_samples = num_samples

    def isValid(self):
        if False:
            return 10
        if self.x < 0 or self.y < 0 or self.x >= self.w or (self.y >= self.h):
            print('Invalid dimensions loc({}, {}), size({}, {})'.format(self.x, self.y, self.w, self.h))
            return False
        if self.num_samples < 1 or self.num_pixels < 1:
            print('Not enough pixels {} or samples {} specified'.format(self.num_pixels, self.num_samples))
            return False
        totalPixels = self.w * self.h
        leftOver = totalPixels - self.w * self.y + self.x
        if leftOver < self.num_pixels:
            print('Too many pixels ({}) specified, for current descriptor at most {} pixels can be rendered'.format(self.num_pixels, leftOver))
            return False
        return True

    def getID(self):
        if False:
            while True:
                i = 10
        return self.id

    def getX(self):
        if False:
            for i in range(10):
                print('nop')
        return self.x

    def getY(self):
        if False:
            return 10
        return self.y

    def getW(self):
        if False:
            i = 10
            return i + 15
        return self.w

    def getH(self):
        if False:
            return 10
        return self.h

    def getNumPixels(self):
        if False:
            for i in range(10):
                print('nop')
        return self.num_pixels

    def getNumSamples(self):
        if False:
            while True:
                i = 10
        return self.num_samples

class RenderTask:

    @classmethod
    def createRenderTask(cls, renderTaskDesc, scene_data, callback):
        if False:
            for i in range(10):
                print('nop')
        if not renderTaskDesc.isValid():
            return None
        try:
            data_stream = StringIO(scene_data)
            camera = Camera(data_stream)
            scene = Scene(data_stream, camera.view_position)
        except Exception as ex:
            print('Failed to read camera or scene from serialized data')
            print(ex)
            return None
        return RenderTask(renderTaskDesc, camera, scene, callback)

    def __init__(self, desc, camera, scene, callback):
        if False:
            i = 10
            return i + 15
        self.desc = desc
        self.camera = camera
        self.scene = scene
        self.callback = callback

    def isValid(self):
        if False:
            i = 10
            return i + 15
        return self.desc.isValid()

    def getDesc(self):
        if False:
            print('Hello World!')
        return self.desc

    def getCamera(self):
        if False:
            return 10
        return self.camera

    def getScene(self):
        if False:
            return 10
        return self.scene

class RenderTaskResult:

    @classmethod
    def createRenderTaskResult(cls, renderTaskDesc, pixelData):
        if False:
            return 10
        if not renderTaskDesc.isValid():
            return None
        lenPixels = len(pixelData)
        if lenPixels % 3 != 0:
            print('Pixel data len not divisible by 3'.format(lenPixels))
            return None
        if lenPixels // 3 != renderTaskDesc.getNumPixels():
            print('Pixel data length {} differs from descriptor data length {}'.format(lenPixels, renderTaskDesc.getNumPixels()))
            return None
        return RenderTaskResult(renderTaskDesc, pixelData)

    def __init__(self, desc, pixelData):
        if False:
            for i in range(10):
                print('nop')
        self.desc = desc
        self.pixelData = pixelData

    def getDesc(self):
        if False:
            for i in range(10):
                print('nop')
        return self.desc

    def get_pixel_data(self):
        if False:
            return 10
        return self.pixelData