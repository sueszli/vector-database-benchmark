import cython

def empty_float():
    if False:
        while True:
            i = 10
    '\n    >>> float()\n    0.0\n    >>> empty_float()\n    0.0\n    '
    x = float()
    return x

def float_conjugate():
    if False:
        i = 10
        return i + 15
    '\n    >>> float_call_conjugate()\n    1.5\n    '
    x = 1.5.conjugate()
    return x

def float_call_conjugate():
    if False:
        while True:
            i = 10
    '\n    >>> float_call_conjugate()\n    1.5\n    '
    x = float(1.5).conjugate()
    return x

def from_int(i):
    if False:
        while True:
            i = 10
    '\n    >>> from_int(0)\n    0.0\n    >>> from_int(1)\n    1.0\n    >>> from_int(-1)\n    -1.0\n    >>> from_int(99)\n    99.0\n    >>> from_int(-99)\n    -99.0\n\n    >>> for exp in (14, 15, 16, 30, 31, 32, 52, 53, 54, 60, 61, 62, 63, 64):\n    ...     for sign in (1, 0, -1):\n    ...         value = (sign or 1) * 2**exp + sign\n    ...         float_value = from_int(value)\n    ...         assert float_value == float(value), "expected %s2**%s+%s == %r, got %r, difference %r" % (\n    ...             \'-\' if sign < 0 else \'\', exp, sign, float(value), float_value, float_value - float(value))\n    '
    return float(i)

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_bytes(s: bytes):
    if False:
        return 10
    '\n    >>> from_bytes(b"123")\n    123.0\n    >>> from_bytes(b"123.25")\n    123.25\n    >>> from_bytes(b"98_5_6.2_1")\n    9856.21\n    >>> from_bytes(b"12_4_131_123123_1893798127398123_19238712_128937198237.8222113_519879812387")\n    1.2413112312318938e+47\n    >>> from_bytes(b"123E100")\n    1.23e+102\n    >>> from_bytes(b"12__._3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12__._3...\n    >>> from_bytes(b"_12.3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ..._12.3...\n    >>> from_bytes(b"12.3_")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12.3_...\n    >>> from_bytes(b"na_n")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...na_n...\n    >>> from_bytes(b"_" * 10000)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...____...\n    >>> from_bytes(None)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError...\n    '
    return float(s)

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_bytes_literals():
    if False:
        while True:
            i = 10
    '\n    >>> from_bytes_literals()\n    (123.0, 123.23, 123.76, 1e+100)\n    '
    return (float(b'123'), float(b'123.23'), float(b'12_3.7_6'), float(b'1e100'))

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_bytearray(s: bytearray):
    if False:
        return 10
    '\n    >>> from_bytearray(bytearray(b"123"))\n    123.0\n    >>> from_bytearray(bytearray(b"123.25"))\n    123.25\n    >>> from_bytearray(bytearray(b"98_5_6.2_1"))\n    9856.21\n    >>> from_bytearray(bytearray(b"12_4_131_123123_1893798127398123_19238712_128937198237.8222113_519879812387"))\n    1.2413112312318938e+47\n    >>> from_bytearray(bytearray(b"123E100"))\n    1.23e+102\n    >>> from_bytearray(bytearray(b"12__._3"))  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12__._3...\n    >>> from_bytearray(bytearray(b"_12.3"))  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ..._12.3...\n    >>> from_bytearray(bytearray(b"12.3_"))  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12.3_...\n    >>> from_bytearray(bytearray(b"in_f"))  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...in_f...\n    >>> from_bytearray(None)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError...\n    '
    return float(s)

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_str(s: 'str'):
    if False:
        return 10
    '\n    >>> from_str("123")\n    123.0\n    >>> from_str("123.25")\n    123.25\n    >>> from_str("3_21.2_5")\n    321.25\n    >>> from_str("12_4_131_123123_1893798127398123_19238712_128937198237.8222113_519879812387")\n    1.2413112312318938e+47\n    >>> from_str("123E100")\n    1.23e+102\n    >>> from_str("12__._3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12__._3...\n    >>> from_str("_12.3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ..._12.3...\n    >>> from_str("12.3_")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12.3_...\n    >>> from_str("n_an")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...n_an...\n    >>> from_str(None)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError...\n    '
    return float(s)

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_str_literals():
    if False:
        return 10
    '\n    >>> from_str_literals()\n    (123.0, 123.23, 124.23, 1e+100)\n    '
    return (float('123'), float('123.23'), float('1_2_4.2_3'), float('1e100'))

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_unicode(s: 'unicode'):
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> from_unicode(u"123")\n    123.0\n    >>> from_unicode(u"123.25")\n    123.25\n    >>> from_unicode(u"12_4.8_5")\n    124.85\n    >>> from_unicode(u"12_4_131_123123_1893798127398123_19238712_128937198237.8222113_519879812387")\n    1.2413112312318938e+47\n    >>> from_unicode(u"123E100")\n    1.23e+102\n    >>> from_unicode("೬")\n    6.0\n    >>> from_unicode(u"123.23\\N{PUNCTUATION SPACE}")\n    123.23\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} 123.23 \\N{PUNCTUATION SPACE}")\n    123.23\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} 12_3.2_3 \\N{PUNCTUATION SPACE}")\n    123.23\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} " * 25 + u"123.54 " + u"\\N{PUNCTUATION SPACE} " * 22)  # >= 40 chars\n    123.54\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} " * 25 + u"1_23.5_4 " + u"\\N{PUNCTUATION SPACE} " * 22)\n    123.54\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} " + u"123.54 " * 2 + u"\\N{PUNCTUATION SPACE}")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...123.54 123.54...\n    >>> from_unicode(u"\\N{PUNCTUATION SPACE} " * 25 + u"123.54 " * 2 + u"\\N{PUNCTUATION SPACE} " * 22)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...123.54 123.54...\n    >>> from_unicode(u"_12__._3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ..._12__._3...\n    >>> from_unicode(u"_12.3")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ..._12.3...\n    >>> from_unicode(u"12.3_")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...12.3_...\n    >>> from_unicode(u"i_nf")  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    ValueError: ...i_nf...\n    >>> from_unicode(None)  # doctest: +ELLIPSIS\n    Traceback (most recent call last):\n    TypeError...\n    '
    return float(s)

@cython.test_assert_path_exists('//CoerceToPyTypeNode', '//CoerceToPyTypeNode//PythonCapiCallNode')
def from_unicode_literals():
    if False:
        return 10
    '\n    >>> from_unicode_literals()\n    (123.0, 123.23, 123.45, 1e+100, 123.23)\n    '
    return (float(u'123'), float(u'123.23'), float(u'12_3.4_5'), float(u'1e100'), float(u'123.23\u2008'))

def catch_valueerror(val):
    if False:
        while True:
            i = 10
    '\n    >>> catch_valueerror("foo")\n    False\n    >>> catch_valueerror(u"foo")\n    False\n    >>> catch_valueerror(b"foo")\n    False\n    >>> catch_valueerror(bytearray(b"foo"))\n    False\n    >>> catch_valueerror("-1")\n    -1.0\n    '
    try:
        return float(val)
    except ValueError:
        return False