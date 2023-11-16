import sys
from enum import IntEnum
from . import Image
try:
    from . import _imagingcms
except ImportError as ex:
    from ._util import DeferredError
    _imagingcms = DeferredError(ex)
DESCRIPTION = '\npyCMS\n\n    a Python / PIL interface to the littleCMS ICC Color Management System\n    Copyright (C) 2002-2003 Kevin Cazabon\n    kevin@cazabon.com\n    https://www.cazabon.com\n\n    pyCMS home page:  https://www.cazabon.com/pyCMS\n    littleCMS home page:  https://www.littlecms.com\n    (littleCMS is Copyright (C) 1998-2001 Marti Maria)\n\n    Originally released under LGPL.  Graciously donated to PIL in\n    March 2009, for distribution under the standard PIL license\n\n    The pyCMS.py module provides a "clean" interface between Python/PIL and\n    pyCMSdll, taking care of some of the more complex handling of the direct\n    pyCMSdll functions, as well as error-checking and making sure that all\n    relevant data is kept together.\n\n    While it is possible to call pyCMSdll functions directly, it\'s not highly\n    recommended.\n\n    Version History:\n\n        1.0.0 pil       Oct 2013 Port to LCMS 2.\n\n        0.1.0 pil mod   March 10, 2009\n\n                        Renamed display profile to proof profile. The proof\n                        profile is the profile of the device that is being\n                        simulated, not the profile of the device which is\n                        actually used to display/print the final simulation\n                        (that\'d be the output profile) - also see LCMSAPI.txt\n                        input colorspace -> using \'renderingIntent\' -> proof\n                        colorspace -> using \'proofRenderingIntent\' -> output\n                        colorspace\n\n                        Added LCMS FLAGS support.\n                        Added FLAGS["SOFTPROOFING"] as default flag for\n                        buildProofTransform (otherwise the proof profile/intent\n                        would be ignored).\n\n        0.1.0 pil       March 2009 - added to PIL, as PIL.ImageCms\n\n        0.0.2 alpha     Jan 6, 2002\n\n                        Added try/except statements around type() checks of\n                        potential CObjects... Python won\'t let you use type()\n                        on them, and raises a TypeError (stupid, if you ask\n                        me!)\n\n                        Added buildProofTransformFromOpenProfiles() function.\n                        Additional fixes in DLL, see DLL code for details.\n\n        0.0.1 alpha     first public release, Dec. 26, 2002\n\n    Known to-do list with current version (of Python interface, not pyCMSdll):\n\n        none\n\n'
VERSION = '1.0.0 pil'
core = _imagingcms

class Intent(IntEnum):
    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3

class Direction(IntEnum):
    INPUT = 0
    OUTPUT = 1
    PROOF = 2
FLAGS = {'MATRIXINPUT': 1, 'MATRIXOUTPUT': 2, 'MATRIXONLY': 1 | 2, 'NOWHITEONWHITEFIXUP': 4, 'NOPRELINEARIZATION': 16, 'GUESSDEVICECLASS': 32, 'NOTCACHE': 64, 'NOTPRECALC': 256, 'NULLTRANSFORM': 512, 'HIGHRESPRECALC': 1024, 'LOWRESPRECALC': 2048, 'WHITEBLACKCOMPENSATION': 8192, 'BLACKPOINTCOMPENSATION': 8192, 'GAMUTCHECK': 4096, 'SOFTPROOFING': 16384, 'PRESERVEBLACK': 32768, 'NODEFAULTRESOURCEDEF': 16777216, 'GRIDPOINTS': lambda n: (n & 255) << 16}
_MAX_FLAG = 0
for flag in FLAGS.values():
    if isinstance(flag, int):
        _MAX_FLAG = _MAX_FLAG | flag

class ImageCmsProfile:

    def __init__(self, profile):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param profile: Either a string representing a filename,\n            a file like object containing a profile or a\n            low-level profile object\n\n        '
        if isinstance(profile, str):
            if sys.platform == 'win32':
                profile_bytes_path = profile.encode()
                try:
                    profile_bytes_path.decode('ascii')
                except UnicodeDecodeError:
                    with open(profile, 'rb') as f:
                        self._set(core.profile_frombytes(f.read()))
                    return
            self._set(core.profile_open(profile), profile)
        elif hasattr(profile, 'read'):
            self._set(core.profile_frombytes(profile.read()))
        elif isinstance(profile, _imagingcms.CmsProfile):
            self._set(profile)
        else:
            msg = 'Invalid type for Profile'
            raise TypeError(msg)

    def _set(self, profile, filename=None):
        if False:
            return 10
        self.profile = profile
        self.filename = filename
        self.product_name = None
        self.product_info = None

    def tobytes(self):
        if False:
            return 10
        '\n        Returns the profile in a format suitable for embedding in\n        saved images.\n\n        :returns: a bytes object containing the ICC profile.\n        '
        return core.profile_tobytes(self.profile)

class ImageCmsTransform(Image.ImagePointHandler):
    """
    Transform.  This can be used with the procedural API, or with the standard
    :py:func:`~PIL.Image.Image.point` method.

    Will return the output profile in the ``output.info['icc_profile']``.
    """

    def __init__(self, input, output, input_mode, output_mode, intent=Intent.PERCEPTUAL, proof=None, proof_intent=Intent.ABSOLUTE_COLORIMETRIC, flags=0):
        if False:
            while True:
                i = 10
        if proof is None:
            self.transform = core.buildTransform(input.profile, output.profile, input_mode, output_mode, intent, flags)
        else:
            self.transform = core.buildProofTransform(input.profile, output.profile, proof.profile, input_mode, output_mode, intent, proof_intent, flags)
        self.input_mode = self.inputMode = input_mode
        self.output_mode = self.outputMode = output_mode
        self.output_profile = output

    def point(self, im):
        if False:
            print('Hello World!')
        return self.apply(im)

    def apply(self, im, imOut=None):
        if False:
            for i in range(10):
                print('nop')
        im.load()
        if imOut is None:
            imOut = Image.new(self.output_mode, im.size, None)
        self.transform.apply(im.im.id, imOut.im.id)
        imOut.info['icc_profile'] = self.output_profile.tobytes()
        return imOut

    def apply_in_place(self, im):
        if False:
            i = 10
            return i + 15
        im.load()
        if im.mode != self.output_mode:
            msg = 'mode mismatch'
            raise ValueError(msg)
        self.transform.apply(im.im.id, im.im.id)
        im.info['icc_profile'] = self.output_profile.tobytes()
        return im

def get_display_profile(handle=None):
    if False:
        print('Hello World!')
    '\n    (experimental) Fetches the profile for the current display device.\n\n    :returns: ``None`` if the profile is not known.\n    '
    if sys.platform != 'win32':
        return None
    from . import ImageWin
    if isinstance(handle, ImageWin.HDC):
        profile = core.get_display_profile_win32(handle, 1)
    else:
        profile = core.get_display_profile_win32(handle or 0)
    if profile is None:
        return None
    return ImageCmsProfile(profile)

class PyCMSError(Exception):
    """(pyCMS) Exception class.
    This is used for all errors in the pyCMS API."""
    pass

def profileToProfile(im, inputProfile, outputProfile, renderingIntent=Intent.PERCEPTUAL, outputMode=None, inPlace=False, flags=0):
    if False:
        return 10
    '\n    (pyCMS) Applies an ICC transformation to a given image, mapping from\n    ``inputProfile`` to ``outputProfile``.\n\n    If the input or output profiles specified are not valid filenames, a\n    :exc:`PyCMSError` will be raised.  If ``inPlace`` is ``True`` and\n    ``outputMode != im.mode``, a :exc:`PyCMSError` will be raised.\n    If an error occurs during application of the profiles,\n    a :exc:`PyCMSError` will be raised.\n    If ``outputMode`` is not a mode supported by the ``outputProfile`` (or by pyCMS),\n    a :exc:`PyCMSError` will be raised.\n\n    This function applies an ICC transformation to im from ``inputProfile``\'s\n    color space to ``outputProfile``\'s color space using the specified rendering\n    intent to decide how to handle out-of-gamut colors.\n\n    ``outputMode`` can be used to specify that a color mode conversion is to\n    be done using these profiles, but the specified profiles must be able\n    to handle that mode.  I.e., if converting im from RGB to CMYK using\n    profiles, the input profile must handle RGB data, and the output\n    profile must handle CMYK data.\n\n    :param im: An open :py:class:`~PIL.Image.Image` object (i.e. Image.new(...)\n        or Image.open(...), etc.)\n    :param inputProfile: String, as a valid filename path to the ICC input\n        profile you wish to use for this image, or a profile object\n    :param outputProfile: String, as a valid filename path to the ICC output\n        profile you wish to use for this image, or a profile object\n    :param renderingIntent: Integer (0-3) specifying the rendering intent you\n        wish to use for the transform\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n        they do.\n    :param outputMode: A valid PIL mode for the output image (i.e. "RGB",\n        "CMYK", etc.).  Note: if rendering the image "inPlace", outputMode\n        MUST be the same mode as the input, or omitted completely.  If\n        omitted, the outputMode will be the same as the mode of the input\n        image (im.mode)\n    :param inPlace: Boolean.  If ``True``, the original image is modified in-place,\n        and ``None`` is returned.  If ``False`` (default), a new\n        :py:class:`~PIL.Image.Image` object is returned with the transform applied.\n    :param flags: Integer (0-...) specifying additional flags\n    :returns: Either None or a new :py:class:`~PIL.Image.Image` object, depending on\n        the value of ``inPlace``\n    :exception PyCMSError:\n    '
    if outputMode is None:
        outputMode = im.mode
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = f'flags must be an integer between 0 and {_MAX_FLAG}'
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        transform = ImageCmsTransform(inputProfile, outputProfile, im.mode, outputMode, renderingIntent, flags=flags)
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v
    return imOut

def getOpenProfile(profileFilename):
    if False:
        for i in range(10):
            print('nop')
    '\n    (pyCMS) Opens an ICC profile file.\n\n    The PyCMSProfile object can be passed back into pyCMS for use in creating\n    transforms and such (as in ImageCms.buildTransformFromOpenProfiles()).\n\n    If ``profileFilename`` is not a valid filename for an ICC profile,\n    a :exc:`PyCMSError` will be raised.\n\n    :param profileFilename: String, as a valid filename path to the ICC profile\n        you wish to open, or a file-like object.\n    :returns: A CmsProfile class object.\n    :exception PyCMSError:\n    '
    try:
        return ImageCmsProfile(profileFilename)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def buildTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent=Intent.PERCEPTUAL, flags=0):
    if False:
        while True:
            i = 10
    '\n    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the\n    ``outputProfile``. Use applyTransform to apply the transform to a given\n    image.\n\n    If the input or output profiles specified are not valid filenames, a\n    :exc:`PyCMSError` will be raised. If an error occurs during creation\n    of the transform, a :exc:`PyCMSError` will be raised.\n\n    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``\n    (or by pyCMS), a :exc:`PyCMSError` will be raised.\n\n    This function builds and returns an ICC transform from the ``inputProfile``\n    to the ``outputProfile`` using the ``renderingIntent`` to determine what to do\n    with out-of-gamut colors.  It will ONLY work for converting images that\n    are in ``inMode`` to images that are in ``outMode`` color format (PIL mode,\n    i.e. "RGB", "RGBA", "CMYK", etc.).\n\n    Building the transform is a fair part of the overhead in\n    ImageCms.profileToProfile(), so if you\'re planning on converting multiple\n    images using the same input/output settings, this can save you time.\n    Once you have a transform object, it can be used with\n    ImageCms.applyProfile() to convert images without the need to re-compute\n    the lookup table for the transform.\n\n    The reason pyCMS returns a class object rather than a handle directly\n    to the transform is that it needs to keep track of the PIL input/output\n    modes that the transform is meant for.  These attributes are stored in\n    the ``inMode`` and ``outMode`` attributes of the object (which can be\n    manually overridden if you really want to, but I don\'t know of any\n    time that would be of use, or would even work).\n\n    :param inputProfile: String, as a valid filename path to the ICC input\n        profile you wish to use for this transform, or a profile object\n    :param outputProfile: String, as a valid filename path to the ICC output\n        profile you wish to use for this transform, or a profile object\n    :param inMode: String, as a valid PIL mode that the appropriate profile\n        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)\n    :param outMode: String, as a valid PIL mode that the appropriate profile\n        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)\n    :param renderingIntent: Integer (0-3) specifying the rendering intent you\n        wish to use for the transform\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n        they do.\n    :param flags: Integer (0-...) specifying additional flags\n    :returns: A CmsTransform class object.\n    :exception PyCMSError:\n    '
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = 'flags must be an integer between 0 and %s' + _MAX_FLAG
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, flags=flags)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def buildProofTransform(inputProfile, outputProfile, proofProfile, inMode, outMode, renderingIntent=Intent.PERCEPTUAL, proofRenderingIntent=Intent.ABSOLUTE_COLORIMETRIC, flags=FLAGS['SOFTPROOFING']):
    if False:
        for i in range(10):
            print('nop')
    '\n    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the\n    ``outputProfile``, but tries to simulate the result that would be\n    obtained on the ``proofProfile`` device.\n\n    If the input, output, or proof profiles specified are not valid\n    filenames, a :exc:`PyCMSError` will be raised.\n\n    If an error occurs during creation of the transform,\n    a :exc:`PyCMSError` will be raised.\n\n    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``\n    (or by pyCMS), a :exc:`PyCMSError` will be raised.\n\n    This function builds and returns an ICC transform from the ``inputProfile``\n    to the ``outputProfile``, but tries to simulate the result that would be\n    obtained on the ``proofProfile`` device using ``renderingIntent`` and\n    ``proofRenderingIntent`` to determine what to do with out-of-gamut\n    colors.  This is known as "soft-proofing".  It will ONLY work for\n    converting images that are in ``inMode`` to images that are in outMode\n    color format (PIL mode, i.e. "RGB", "RGBA", "CMYK", etc.).\n\n    Usage of the resulting transform object is exactly the same as with\n    ImageCms.buildTransform().\n\n    Proof profiling is generally used when using an output device to get a\n    good idea of what the final printed/displayed image would look like on\n    the ``proofProfile`` device when it\'s quicker and easier to use the\n    output device for judging color.  Generally, this means that the\n    output device is a monitor, or a dye-sub printer (etc.), and the simulated\n    device is something more expensive, complicated, or time consuming\n    (making it difficult to make a real print for color judgement purposes).\n\n    Soft-proofing basically functions by adjusting the colors on the\n    output device to match the colors of the device being simulated. However,\n    when the simulated device has a much wider gamut than the output\n    device, you may obtain marginal results.\n\n    :param inputProfile: String, as a valid filename path to the ICC input\n        profile you wish to use for this transform, or a profile object\n    :param outputProfile: String, as a valid filename path to the ICC output\n        (monitor, usually) profile you wish to use for this transform, or a\n        profile object\n    :param proofProfile: String, as a valid filename path to the ICC proof\n        profile you wish to use for this transform, or a profile object\n    :param inMode: String, as a valid PIL mode that the appropriate profile\n        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)\n    :param outMode: String, as a valid PIL mode that the appropriate profile\n        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)\n    :param renderingIntent: Integer (0-3) specifying the rendering intent you\n        wish to use for the input->proof (simulated) transform\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n        they do.\n    :param proofRenderingIntent: Integer (0-3) specifying the rendering intent\n        you wish to use for proof->output transform\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n        they do.\n    :param flags: Integer (0-...) specifying additional flags\n    :returns: A CmsTransform class object.\n    :exception PyCMSError:\n    '
    if not isinstance(renderingIntent, int) or not 0 <= renderingIntent <= 3:
        msg = 'renderingIntent must be an integer between 0 and 3'
        raise PyCMSError(msg)
    if not isinstance(flags, int) or not 0 <= flags <= _MAX_FLAG:
        msg = 'flags must be an integer between 0 and %s' + _MAX_FLAG
        raise PyCMSError(msg)
    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        if not isinstance(proofProfile, ImageCmsProfile):
            proofProfile = ImageCmsProfile(proofProfile)
        return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, proofProfile, proofRenderingIntent, flags)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v
buildTransformFromOpenProfiles = buildTransform
buildProofTransformFromOpenProfiles = buildProofTransform

def applyTransform(im, transform, inPlace=False):
    if False:
        print('Hello World!')
    "\n    (pyCMS) Applies a transform to a given image.\n\n    If ``im.mode != transform.inMode``, a :exc:`PyCMSError` is raised.\n\n    If ``inPlace`` is ``True`` and ``transform.inMode != transform.outMode``, a\n    :exc:`PyCMSError` is raised.\n\n    If ``im.mode``, ``transform.inMode`` or ``transform.outMode`` is not\n    supported by pyCMSdll or the profiles you used for the transform, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while the transform is being applied,\n    a :exc:`PyCMSError` is raised.\n\n    This function applies a pre-calculated transform (from\n    ImageCms.buildTransform() or ImageCms.buildTransformFromOpenProfiles())\n    to an image. The transform can be used for multiple images, saving\n    considerable calculation time if doing the same conversion multiple times.\n\n    If you want to modify im in-place instead of receiving a new image as\n    the return value, set ``inPlace`` to ``True``.  This can only be done if\n    ``transform.inMode`` and ``transform.outMode`` are the same, because we can't\n    change the mode in-place (the buffer sizes for some modes are\n    different).  The default behavior is to return a new :py:class:`~PIL.Image.Image`\n    object of the same dimensions in mode ``transform.outMode``.\n\n    :param im: An :py:class:`~PIL.Image.Image` object, and im.mode must be the same\n        as the ``inMode`` supported by the transform.\n    :param transform: A valid CmsTransform class object\n    :param inPlace: Bool.  If ``True``, ``im`` is modified in place and ``None`` is\n        returned, if ``False``, a new :py:class:`~PIL.Image.Image` object with the\n        transform applied is returned (and ``im`` is not changed). The default is\n        ``False``.\n    :returns: Either ``None``, or a new :py:class:`~PIL.Image.Image` object,\n        depending on the value of ``inPlace``. The profile will be returned in\n        the image's ``info['icc_profile']``.\n    :exception PyCMSError:\n    "
    try:
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v
    return imOut

def createProfile(colorSpace, colorTemp=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    (pyCMS) Creates a profile.\n\n    If colorSpace not in ``["LAB", "XYZ", "sRGB"]``,\n    a :exc:`PyCMSError` is raised.\n\n    If using LAB and ``colorTemp`` is not a positive integer,\n    a :exc:`PyCMSError` is raised.\n\n    If an error occurs while creating the profile,\n    a :exc:`PyCMSError` is raised.\n\n    Use this function to create common profiles on-the-fly instead of\n    having to supply a profile on disk and knowing the path to it.  It\n    returns a normal CmsProfile object that can be passed to\n    ImageCms.buildTransformFromOpenProfiles() to create a transform to apply\n    to images.\n\n    :param colorSpace: String, the color space of the profile you wish to\n        create.\n        Currently only "LAB", "XYZ", and "sRGB" are supported.\n    :param colorTemp: Positive integer for the white point for the profile, in\n        degrees Kelvin (i.e. 5000, 6500, 9600, etc.).  The default is for D50\n        illuminant if omitted (5000k).  colorTemp is ONLY applied to LAB\n        profiles, and is ignored for XYZ and sRGB.\n    :returns: A CmsProfile class object\n    :exception PyCMSError:\n    '
    if colorSpace not in ['LAB', 'XYZ', 'sRGB']:
        msg = f'Color space not supported for on-the-fly profile creation ({colorSpace})'
        raise PyCMSError(msg)
    if colorSpace == 'LAB':
        try:
            colorTemp = float(colorTemp)
        except (TypeError, ValueError) as e:
            msg = f'Color temperature must be numeric, "{colorTemp}" not valid'
            raise PyCMSError(msg) from e
    try:
        return core.createProfile(colorSpace, colorTemp)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileName(profile):
    if False:
        while True:
            i = 10
    "\n\n    (pyCMS) Gets the internal product name for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile,\n    a :exc:`PyCMSError` is raised If an error occurs while trying\n    to obtain the name tag, a :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the INTERNAL name of the profile (stored\n    in an ICC tag in the profile itself), usually the one used when the\n    profile was originally created.  Sometimes this tag also contains\n    additional information supplied by the creator.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal name of the profile as stored\n        in an ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        model = profile.profile.model
        manufacturer = profile.profile.manufacturer
        if not (model or manufacturer):
            return (profile.profile.profile_description or '') + '\n'
        if not manufacturer or len(model) > 30:
            return model + '\n'
        return f'{model} - {manufacturer}\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileInfo(profile):
    if False:
        return 10
    "\n    (pyCMS) Gets the internal product information for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile,\n    a :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the info tag,\n    a :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the information stored in the profile's\n    info tag.  This often contains details about the profile, and how it\n    was created, as supplied by the creator.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal profile information stored in\n        an ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        description = profile.profile.profile_description
        cpright = profile.profile.copyright
        arr = []
        for elt in (description, cpright):
            if elt:
                arr.append(elt)
        return '\r\n\r\n'.join(arr) + '\r\n\r\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileCopyright(profile):
    if False:
        return 10
    "\n    (pyCMS) Gets the copyright for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the copyright tag,\n    a :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the information stored in the profile's\n    copyright tag.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal profile information stored in\n        an ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.copyright or '') + '\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileManufacturer(profile):
    if False:
        for i in range(10):
            print('nop')
    "\n    (pyCMS) Gets the manufacturer for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the manufacturer tag, a\n    :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the information stored in the profile's\n    manufacturer tag.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal profile information stored in\n        an ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.manufacturer or '') + '\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileModel(profile):
    if False:
        print('Hello World!')
    "\n    (pyCMS) Gets the model for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the model tag,\n    a :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the information stored in the profile's\n    model tag.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal profile information stored in\n        an ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.model or '') + '\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getProfileDescription(profile):
    if False:
        while True:
            i = 10
    "\n    (pyCMS) Gets the description for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the description tag,\n    a :exc:`PyCMSError` is raised.\n\n    Use this function to obtain the information stored in the profile's\n    description tag.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: A string containing the internal profile information stored in an\n        ICC tag.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.profile_description or '') + '\n'
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def getDefaultIntent(profile):
    if False:
        return 10
    "\n    (pyCMS) Gets the default intent name for the given profile.\n\n    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a\n    :exc:`PyCMSError` is raised.\n\n    If an error occurs while trying to obtain the default intent, a\n    :exc:`PyCMSError` is raised.\n\n    Use this function to determine the default (and usually best optimized)\n    rendering intent for this profile.  Most profiles support multiple\n    rendering intents, but are intended mostly for one type of conversion.\n    If you wish to use a different intent than returned, use\n    ImageCms.isIntentSupported() to verify it will work first.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :returns: Integer 0-3 specifying the default rendering intent for this\n        profile.\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n            they do.\n    :exception PyCMSError:\n    "
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return profile.profile.rendering_intent
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def isIntentSupported(profile, intent, direction):
    if False:
        i = 10
        return i + 15
    '\n    (pyCMS) Checks if a given intent is supported.\n\n    Use this function to verify that you can use your desired\n    ``intent`` with ``profile``, and that ``profile`` can be used for the\n    input/output/proof profile as you desire.\n\n    Some profiles are created specifically for one "direction", can cannot\n    be used for others. Some profiles can only be used for certain\n    rendering intents, so it\'s best to either verify this before trying\n    to create a transform with them (using this function), or catch the\n    potential :exc:`PyCMSError` that will occur if they don\'t\n    support the modes you select.\n\n    :param profile: EITHER a valid CmsProfile object, OR a string of the\n        filename of an ICC profile.\n    :param intent: Integer (0-3) specifying the rendering intent you wish to\n        use with this profile\n\n            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)\n            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1\n            ImageCms.Intent.SATURATION            = 2\n            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3\n\n        see the pyCMS documentation for details on rendering intents and what\n            they do.\n    :param direction: Integer specifying if the profile is to be used for\n        input, output, or proof\n\n            INPUT  = 0 (or use ImageCms.Direction.INPUT)\n            OUTPUT = 1 (or use ImageCms.Direction.OUTPUT)\n            PROOF  = 2 (or use ImageCms.Direction.PROOF)\n\n    :returns: 1 if the intent/direction are supported, -1 if they are not.\n    :exception PyCMSError:\n    '
    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        if profile.profile.is_intent_supported(intent, direction):
            return 1
        else:
            return -1
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

def versions():
    if False:
        while True:
            i = 10
    '\n    (pyCMS) Fetches versions.\n    '
    return (VERSION, core.littlecms_version, sys.version.split()[0], Image.__version__)