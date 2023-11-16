import numpy as np
try:
    import cv2
    headblur_possible = True
    if cv2.__version__ >= '3.0.0':
        cv2.CV_AA = cv2.LINE_AA
except Exception:
    headblur_possible = False

def headblur(clip, fx, fy, radius, intensity=None):
    if False:
        i = 10
        return i + 15
    'Returns a filter that will blur a moving part (a head ?) of the frames.\n\n    The position of the blur at time t is defined by (fx(t), fy(t)), the radius\n    of the blurring by ``radius`` and the intensity of the blurring by ``intensity``.\n\n    Requires OpenCV for the circling and the blurring. Automatically deals with the\n    case where part of the image goes offscreen.\n    '
    if intensity is None:
        intensity = int(2 * radius / 3)

    def filter(gf, t):
        if False:
            print('Hello World!')
        im = gf(t).copy()
        (h, w, d) = im.shape
        (x, y) = (int(fx(t)), int(fy(t)))
        (x1, x2) = (max(0, x - radius), min(x + radius, w))
        (y1, y2) = (max(0, y - radius), min(y + radius, h))
        region_size = (y2 - y1, x2 - x1)
        mask = np.zeros(region_size).astype('uint8')
        cv2.circle(mask, (radius, radius), radius, 255, -1, lineType=cv2.CV_AA)
        mask = np.dstack(3 * [1.0 / 255 * mask])
        orig = im[y1:y2, x1:x2]
        blurred = cv2.blur(orig, (intensity, intensity))
        im[y1:y2, x1:x2] = mask * blurred + (1 - mask) * orig
        return im
    return clip.transform(filter)
if not headblur_possible:
    doc = headblur.__doc__

    def headblur(clip, fx, fy, r_zone, r_blur=None):
        if False:
            for i in range(10):
                print('nop')
        'Fallback headblur FX function, used if OpenCV is not installed.\n\n        This docstring will be replaced at runtime.\n        '
        raise IOError('fx painting needs opencv')
    headblur.__doc__ = doc