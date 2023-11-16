from textual._line_split import line_split

def test_split_string_to_lines_and_endings():
    if False:
        for i in range(10):
            print('nop')
    assert line_split('Hello\r\nWorld\n') == [('Hello', '\r\n'), ('World', '\n')]
    assert line_split('Hello\rWorld\r\n') == [('Hello', '\r'), ('World', '\r\n')]
    assert line_split('Hello\nWorld\r') == [('Hello', '\n'), ('World', '\r')]
    assert line_split('Hello World') == [('Hello World', '')]
    assert line_split('') == []
    assert line_split('Hello\nWorld\nHow\nAre\nYou\n') == [('Hello', '\n'), ('World', '\n'), ('How', '\n'), ('Are', '\n'), ('You', '\n')]
    assert line_split('a') == [('a', '')]