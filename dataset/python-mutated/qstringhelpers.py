"""QString compatibility."""

def qstring_length(text):
    if False:
        while True:
            i = 10
    '\n    Tries to compute what the length of an utf16-encoded QString would be.\n    '
    utf16_text = text.encode('utf16')
    length = len(utf16_text) // 2
    if utf16_text[:2] in [b'\xff\xfe', b'\xff\xff', b'\xfe\xff']:
        length -= 1
    return length