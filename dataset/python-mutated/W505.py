"""Here's a top-level docstring that's over the limit."""

def f1():
    if False:
        return 10
    "Here's a docstring that's also over the limit."
    x = 1
    x = 2
    print("Here's a string that's over the limit, but it's not a docstring.")
'This is also considered a docstring, and is over the limit.'

def f2():
    if False:
        return 10
    "Here's a multi-line docstring.\n\n    It's over the limit on this line, which isn't the first line in the docstring.\n    "

def f3():
    if False:
        return 10
    "Here's a multi-line docstring.\n\n    It's over the limit on this line, which isn't the first line in the docstring."