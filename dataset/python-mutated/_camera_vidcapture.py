"""pygame.camera.Camera implementation using the videocapture module for windows.

http://videocapture.sourceforge.net/

Binary windows wheels:
  https://www.lfd.uci.edu/~gohlke/pythonlibs/#videocapture
"""
import pygame

def list_cameras():
    if False:
        while True:
            i = 10
    'Always only lists one camera.\n\n    Functionality not supported in videocapture module.\n    '
    return [0]

def init():
    if False:
        return 10
    global vidcap
    try:
        import vidcap as vc
    except ImportError:
        from VideoCapture import vidcap as vc
    vidcap = vc

def quit():
    if False:
        print('Hello World!')
    global vidcap
    vidcap = None

class Camera:

    def __init__(self, device=0, size=(640, 480), mode='RGB', show_video_window=0):
        if False:
            i = 10
            return i + 15
        'device:  VideoCapture enumerates the available video capture devices\n                 on your system.  If you have more than one device, specify\n                 the desired one here.  The device number starts from 0.\n\n        show_video_window: 0 ... do not display a video window (the default)\n                           1 ... display a video window\n\n                         Mainly used for debugging, since the video window\n                         can not be closed or moved around.\n        '
        self.dev = vidcap.new_Dev(device, show_video_window)
        (width, height) = size
        self.dev.setresolution(width, height)

    def display_capture_filter_properties(self):
        if False:
            while True:
                i = 10
        'Displays a dialog containing the property page of the capture filter.\n\n        For VfW drivers you may find the option to select the resolution most\n        likely here.\n        '
        self.dev.displaycapturefilterproperties()

    def display_capture_pin_properties(self):
        if False:
            i = 10
            return i + 15
        'Displays a dialog containing the property page of the capture pin.\n\n        For WDM drivers you may find the option to select the resolution most\n        likely here.\n        '
        self.dev.displaycapturepinproperties()

    def set_resolution(self, width, height):
        if False:
            while True:
                i = 10
        'Sets the capture resolution. (without dialog)'
        self.dev.setresolution(width, height)

    def get_buffer(self):
        if False:
            print('Hello World!')
        'Returns a string containing the raw pixel data.'
        return self.dev.getbuffer()

    def start(self):
        if False:
            return 10
        'Not implemented.'

    def set_controls(self, **kwargs):
        if False:
            print('Hello World!')
        'Not implemented.'

    def stop(self):
        if False:
            print('Hello World!')
        'Not implemented.'

    def get_image(self, dest_surf=None):
        if False:
            print('Hello World!')
        ' '
        return self.get_surface(dest_surf)

    def get_surface(self, dest_surf=None):
        if False:
            i = 10
            return i + 15
        'Returns a pygame Surface.'
        (abuffer, width, height) = self.get_buffer()
        if not abuffer:
            return None
        surf = pygame.image.frombuffer(abuffer, (width, height), 'BGR')
        surf = pygame.transform.flip(surf, 0, 1)
        if dest_surf:
            dest_surf.blit(surf, (0, 0))
        else:
            dest_surf = surf
        return dest_surf
if __name__ == '__main__':
    import pygame.examples.camera
    pygame.camera.Camera = Camera
    pygame.camera.list_cameras = list_cameras
    pygame.examples.camera.main()