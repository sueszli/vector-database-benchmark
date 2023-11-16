from textual.widgets.text_area import Document
TEXT = 'I must not fear.\nFear is the mind-killer.'

def test_insert_no_newlines():
    if False:
        return 10
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), ' really')
    assert document.lines == ['I really must not fear.', 'Fear is the mind-killer.']

def test_insert_empty_string():
    if False:
        i = 10
        return i + 15
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), '')
    assert document.lines == ['I must not fear.', 'Fear is the mind-killer.']

def test_insert_invalid_column():
    if False:
        while True:
            i = 10
    document = Document(TEXT)
    document.replace_range((0, 999), (0, 999), ' really')
    assert document.lines == ['I must not fear. really', 'Fear is the mind-killer.']

def test_insert_invalid_row_and_column():
    if False:
        while True:
            i = 10
    document = Document(TEXT)
    document.replace_range((999, 0), (999, 0), ' really')
    assert document.lines == ['I must not fear.', 'Fear is the mind-killer.', ' really']

def test_insert_range_newline_file_start():
    if False:
        for i in range(10):
            print('nop')
    document = Document(TEXT)
    document.replace_range((0, 0), (0, 0), '\n')
    assert document.lines == ['', 'I must not fear.', 'Fear is the mind-killer.']

def test_insert_newline_splits_line():
    if False:
        while True:
            i = 10
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), '\n')
    assert document.lines == ['I', ' must not fear.', 'Fear is the mind-killer.']

def test_insert_newline_splits_line_selection():
    if False:
        print('Hello World!')
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 6), '\n')
    assert document.lines == ['I', ' not fear.', 'Fear is the mind-killer.']

def test_insert_multiple_lines_ends_with_newline():
    if False:
        while True:
            i = 10
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), 'Hello,\nworld!\n')
    assert document.lines == ['IHello,', 'world!', ' must not fear.', 'Fear is the mind-killer.']

def test_insert_multiple_lines_ends_with_no_newline():
    if False:
        i = 10
        return i + 15
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), 'Hello,\nworld!')
    assert document.lines == ['IHello,', 'world! must not fear.', 'Fear is the mind-killer.']

def test_insert_multiple_lines_starts_with_newline():
    if False:
        while True:
            i = 10
    document = Document(TEXT)
    document.replace_range((0, 1), (0, 1), '\nHello,\nworld!\n')
    assert document.lines == ['I', 'Hello,', 'world!', ' must not fear.', 'Fear is the mind-killer.']

def test_insert_range_text_no_newlines():
    if False:
        while True:
            i = 10
    'Ensuring we can do a simple replacement of text.'
    document = Document(TEXT)
    document.replace_range((0, 2), (0, 6), 'MUST')
    assert document.lines == ['I MUST not fear.', 'Fear is the mind-killer.']
TEXT_NEWLINE_EOF = 'I must not fear.\nFear is the mind-killer.\n'

def test_newline_eof():
    if False:
        i = 10
        return i + 15
    document = Document(TEXT_NEWLINE_EOF)
    assert document.lines == ['I must not fear.', 'Fear is the mind-killer.', '']