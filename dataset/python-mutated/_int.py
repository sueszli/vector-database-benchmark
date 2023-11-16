from __future__ import unicode_literals, division, absolute_import, print_function

def fill_width(bytes_, width):
    if False:
        return 10
    '\n    Ensure a byte string representing a positive integer is a specific width\n    (in bytes)\n\n    :param bytes_:\n        The integer byte string\n\n    :param width:\n        The desired width as an integer\n\n    :return:\n        A byte string of the width specified\n    '
    while len(bytes_) < width:
        bytes_ = b'\x00' + bytes_
    return bytes_