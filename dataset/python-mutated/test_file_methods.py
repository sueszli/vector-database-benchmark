"""Methods of File Objects

@see: https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects

Reading from a file does not always have to be sequential. There are methods to look for
specific locations in the file, much like flipping to a page in a book.
"""

def test_file_methods():
    if False:
        for i in range(10):
            print('nop')
    'Methods of File Objects'
    multi_line_file = open('src/files/multi_line_file.txt', 'r')
    binary_file = open('src/files/binary_file', 'r')
    read_data = multi_line_file.read()
    assert read_data == 'first line\nsecond line\nthird line'
    assert binary_file.seek(0) == 0
    assert binary_file.seek(6) == 6
    assert binary_file.read(1) == '6'
    multi_line_file.seek(0)
    assert multi_line_file.readline() == 'first line\n'
    assert multi_line_file.readline() == 'second line\n'
    assert multi_line_file.readline() == 'third line'
    assert multi_line_file.readline() == ''
    multi_line_file.close()
    binary_file.close()