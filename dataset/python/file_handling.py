"""
File handling is a fundamental concept in Python that involves
opening, reading, writing, and appending to files. This module
demonstrates the basics of file handling in Python.

Python provides various ways to work with files. We can use the
builtin 'open' function to open files in different modes like
reading ('r'), writing ('w'), and appending ('a').
"""
import os

_TARGET_FILE = "sample.txt"


def read_file(filename):
    """Read content from existing file."""
    with open(filename, "r") as file:
        content = file.read()
        return content


def write_file(filename, content):
    """Write content to new file."""
    with open(filename, "w") as file:
        file.write(content)
    return f"Content written to '{filename}'."


def append_file(filename, content):
    """Append content to existing file."""
    with open(filename, "a") as file:
        file.write(content)
    return f"Content appended to '{filename}'."


def delete_file(filename):
    """Delete content of existing file."""
    os.remove(filename)
    return f"'{filename}' has been deleted."


def main():
    result = write_file(_TARGET_FILE, "This is a test.")
    assert result == f"Content written to '{_TARGET_FILE}'."

    content = read_file(_TARGET_FILE)
    assert content == "This is a test."

    append_result = append_file(_TARGET_FILE, "\nThis is an appended line.")
    assert append_result == f"Content appended to '{_TARGET_FILE}'."

    content = read_file(_TARGET_FILE)
    assert content == "This is a test.\nThis is an appended line."

    delete_result = delete_file(_TARGET_FILE)
    assert delete_result == f"'{_TARGET_FILE}' has been deleted."


if __name__ == "__main__":
    main()
