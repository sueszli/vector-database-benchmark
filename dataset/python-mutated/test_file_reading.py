"""Reading and Writing Files

@see: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files

The process of reading and writing to a file is like finding a book and opening a book.
First, the file is located, opened to the first page, then reading/writing begins until it reaches
the end of the file.
"""

def test_files_open():
    if False:
        return 10
    "Open files\n\n    open() returns a file object, and is most commonly used with two arguments:\n    open(filename, mode).\n\n    The first argument is a string containing the filename. The second argument is another string\n    containing a few characters describing the way in which the file will be used. mode can be:\n\n    - 'r' when the file will only be read,\n    - 'w' for only writing (an existing file with the same name will be erased),\n    - 'a' opens the file for appending; any data written to the file is automatically added to end.\n    - 'r+' opens the file for both reading and writing.\n\n    The mode argument is optional; 'r' will be assumed if it’s omitted.\n\n    Normally, files are opened in text mode, that means, you read and write strings from and to the\n    file, which are encoded in a specific encoding. If encoding is not specified, the default is\n    platform dependent (see open()). 'b' appended to the mode opens the file in binary mode: now\n    the data is read and written in the form of bytes objects. This mode should be used for all\n    files that don’t contain text.\n\n    In text mode, the default when reading is to convert platform-specific line endings (\n on\n    Unix, \r\n on Windows) to just \n. When writing in text mode, the default is to convert\n    occurrences of \n back to platform-specific line endings. This behind-the-scenes modification\n    to file data is fine for text files, but will corrupt binary data like that in JPEG or EXE\n    files. Be very careful to use binary mode when reading and writing such files.\n\n    It is good practice to use the with keyword when dealing with file objects. The advantage is\n    that the file is properly closed after its suite finishes, even if an exception is raised at\n    some point. Using with is also much shorter than writing equivalent try-finally blocks:\n    "
    file = open('src/files/multi_line_file.txt', 'r')
    assert not file.closed
    read_data = file.read()
    assert read_data == 'first line\nsecond line\nthird line'
    file.close()
    assert file.closed
    with open('src/files/multi_line_file.txt', 'r') as file:
        read_data = file.read()
        assert read_data == 'first line\nsecond line\nthird line'
    assert file.closed