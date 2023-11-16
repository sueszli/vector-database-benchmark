"""
Module for controlling the LED matrix or reading environment data on the SenseHat of a Raspberry Pi.

.. versionadded:: 2017.7.0

:maintainer:    Benedikt Werner <1benediktwerner@gmail.com>, Joachim Werner <joe@suse.com>
:maturity:      new
:depends:       sense_hat Python module

The rotation of the Pi can be specified in a pillar.
This is useful if the Pi is used upside down or sideways to correct the orientation of the image being shown.

Example:

.. code-block:: yaml

    sensehat:
        rotation: 90

"""
import logging
try:
    from sense_hat import SenseHat
    has_sense_hat = True
except (ImportError, NameError):
    _sensehat = None
    has_sense_hat = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load the module if SenseHat is available\n    '
    if has_sense_hat:
        try:
            _sensehat = SenseHat()
        except OSError:
            return (False, 'This module can only be used on a Raspberry Pi with a SenseHat.')
        rotation = __salt__['pillar.get']('sensehat:rotation', 0)
        if rotation in [0, 90, 180, 270]:
            _sensehat.set_rotation(rotation, False)
        else:
            log.error('%s is not a valid rotation. Using default rotation.', rotation)
        return True
    return (False, "The SenseHat execution module cannot be loaded: 'sense_hat' python library unavailable.")

def set_pixels(pixels):
    if False:
        return 10
    '\n    Sets the entire LED matrix based on a list of 64 pixel values\n\n    pixels\n        A list of 64 ``[R, G, B]`` color values.\n    '
    _sensehat.set_pixels(pixels)
    return {'pixels': pixels}

def get_pixels():
    if False:
        return 10
    "\n    Returns a list of 64 smaller lists of ``[R, G, B]`` pixels representing the\n    the currently displayed image on the LED matrix.\n\n    .. note::\n        When using ``set_pixels`` the pixel values can sometimes change when\n        you read them again using ``get_pixels``. This is because we specify each\n        pixel element as 8 bit numbers (0 to 255) but when they're passed into the\n        Linux frame buffer for the LED matrix the numbers are bit shifted down\n        to fit into RGB 565. 5 bits for red, 6 bits for green and 5 bits for blue.\n        The loss of binary precision when performing this conversion\n        (3 bits lost for red, 2 for green and 3 for blue) accounts for the\n        discrepancies you see.\n\n        The ``get_pixels`` method provides an accurate representation of how the\n        pixels end up in frame buffer memory after you have called ``set_pixels``.\n    "
    return _sensehat.get_pixels()

def set_pixel(x, y, color):
    if False:
        i = 10
        return i + 15
    "\n    Sets a single pixel on the LED matrix to a specified color.\n\n    x\n        The x coordinate of the pixel. Ranges from 0 on the left to 7 on the right.\n    y\n        The y coordinate of the pixel. Ranges from 0 at the top to 7 at the bottom.\n    color\n        The new color of the pixel as a list of ``[R, G, B]`` values.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.set_pixel 0 0 '[255, 0, 0]'\n    "
    _sensehat.set_pixel(x, y, color)
    return {'color': color}

def get_pixel(x, y):
    if False:
        i = 10
        return i + 15
    '\n    Returns the color of a single pixel on the LED matrix.\n\n    x\n        The x coordinate of the pixel. Ranges from 0 on the left to 7 on the right.\n    y\n        The y coordinate of the pixel. Ranges from 0 at the top to 7 at the bottom.\n\n    .. note::\n        Please read the note for ``get_pixels``\n    '
    return _sensehat.get_pixel(x, y)

def low_light(low_light=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets the LED matrix to low light mode. Useful in a dark environment.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.low_light\n        salt 'raspberry' sensehat.low_light False\n    "
    _sensehat.low_light = low_light
    return {'low_light': low_light}

def show_message(message, msg_type=None, text_color=None, back_color=None, scroll_speed=0.1):
    if False:
        i = 10
        return i + 15
    "\n    Displays a message on the LED matrix.\n\n    message\n        The message to display\n    msg_type\n        The type of the message. Changes the appearance of the message.\n\n        Available types are::\n\n            error:      red text\n            warning:    orange text\n            success:    green text\n            info:       blue text\n\n    scroll_speed\n        The speed at which the message moves over the LED matrix.\n        This value represents the time paused for between shifting the text\n        to the left by one column of pixels. Defaults to '0.1'.\n    text_color\n        The color in which the message is shown. Defaults to '[255, 255, 255]' (white).\n    back_color\n        The background color of the display. Defaults to '[0, 0, 0]' (black).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.show_message 'Status ok'\n        salt 'raspberry' sensehat.show_message 'Something went wrong' error\n        salt 'raspberry' sensehat.show_message 'Red' text_color='[255, 0, 0]'\n        salt 'raspberry' sensehat.show_message 'Hello world' None '[0, 0, 255]' '[255, 255, 0]' 0.2\n    "
    text_color = text_color or [255, 255, 255]
    back_color = back_color or [0, 0, 0]
    color_by_type = {'error': [255, 0, 0], 'warning': [255, 100, 0], 'success': [0, 255, 0], 'info': [0, 0, 255]}
    if msg_type in color_by_type:
        text_color = color_by_type[msg_type]
    _sensehat.show_message(message, scroll_speed, text_color, back_color)
    return {'message': message}

def show_letter(letter, text_color=None, back_color=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Displays a single letter on the LED matrix.\n\n    letter\n        The letter to display\n    text_color\n        The color in which the letter is shown. Defaults to '[255, 255, 255]' (white).\n    back_color\n        The background color of the display. Defaults to '[0, 0, 0]' (black).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.show_letter O\n        salt 'raspberry' sensehat.show_letter X '[255, 0, 0]'\n        salt 'raspberry' sensehat.show_letter B '[0, 0, 255]' '[255, 255, 0]'\n    "
    text_color = text_color or [255, 255, 255]
    back_color = back_color or [0, 0, 0]
    _sensehat.show_letter(letter, text_color, back_color)
    return {'letter': letter}

def show_image(image):
    if False:
        for i in range(10):
            print('nop')
    "\n    Displays an 8 x 8 image on the LED matrix.\n\n    image\n        The path to the image to display. The image must be 8 x 8 pixels in size.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.show_image /tmp/my_image.png\n    "
    return _sensehat.load_image(image)

def clear(color=None):
    if False:
        while True:
            i = 10
    "\n    Sets the LED matrix to a single color or turns all LEDs off.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'raspberry' sensehat.clear\n        salt 'raspberry' sensehat.clear '[255, 0, 0]'\n    "
    if color is None:
        _sensehat.clear()
    else:
        _sensehat.clear(color)
    return {'color': color}

def get_humidity():
    if False:
        while True:
            i = 10
    '\n    Get the percentage of relative humidity from the humidity sensor.\n    '
    return _sensehat.get_humidity()

def get_pressure():
    if False:
        while True:
            i = 10
    '\n    Gets the current pressure in Millibars from the pressure sensor.\n    '
    return _sensehat.get_pressure()

def get_temperature():
    if False:
        return 10
    '\n    Gets the temperature in degrees Celsius from the humidity sensor.\n    Equivalent to calling ``get_temperature_from_humidity``.\n\n    If you get strange results try using ``get_temperature_from_pressure``.\n    '
    return _sensehat.get_temperature()

def get_temperature_from_humidity():
    if False:
        print('Hello World!')
    '\n    Gets the temperature in degrees Celsius from the humidity sensor.\n    '
    return _sensehat.get_temperature_from_humidity()

def get_temperature_from_pressure():
    if False:
        return 10
    '\n    Gets the temperature in degrees Celsius from the pressure sensor.\n    '
    return _sensehat.get_temperature_from_pressure()