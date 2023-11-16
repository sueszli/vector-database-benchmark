"""Documentation Strings.

@see: https://docs.python.org/3/tutorial/controlflow.html#documentation-strings

Here are some conventions about the content and formatting of documentation strings.

The first line should always be a short, concise summary of the object’s purpose. For brevity,
it should not explicitly state the object’s name or type, since these are available by other means
(except if the name happens to be a verb describing a function’s operation). This line should begin
with a capital letter and end with a period.

If there are more lines in the documentation string, the second line should be blank, visually
separating the summary from the rest of the description. The following lines should be one or more
paragraphs describing the object’s calling conventions, its side effects, etc.
"""

def do_nothing():
    if False:
        print('Hello World!')
    "Do nothing, but document it.\n\n    No, really, it doesn't do anything.\n    "
    pass

def test_function_documentation_string():
    if False:
        while True:
            i = 10
    'Test documentation string.'
    assert do_nothing.__doc__ == "Do nothing, but document it.\n\n    No, really, it doesn't do anything.\n    "