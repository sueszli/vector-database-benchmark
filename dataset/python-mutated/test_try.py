"""TRY statement

@see: https://www.w3schools.com/python/python_try_except.asp

"try" statement is used for exception handling.
When an error occurs, or exception as we call it, Python will normally stop and generate an error
message. These exceptions can be handled using the try statement.

The "try" block lets you test a block of code for errors.
The "except" block lets you handle the error.
The "else" block lets you execute the code if no errors were raised.
The "finally" block lets you execute code, regardless of the result of the try- and except blocks.
"""

def test_try():
    if False:
        i = 10
        return i + 15
    'TRY statement'
    exception_has_been_caught = False
    try:
        print(not_existing_variable)
    except NameError:
        exception_has_been_caught = True
    assert exception_has_been_caught
    exception_message = ''
    try:
        print(not_existing_variable)
    except NameError:
        exception_message = 'Variable is not defined'
    assert exception_message == 'Variable is not defined'
    message = ''
    try:
        message += 'Success.'
    except NameError:
        message += 'Something went wrong.'
    else:
        message += 'Nothing went wrong.'
    assert message == 'Success.Nothing went wrong.'
    message = ''
    try:
        print(not_existing_variable)
    except NameError:
        message += 'Something went wrong.'
    finally:
        message += 'The "try except" is finished.'
    assert message == 'Something went wrong.The "try except" is finished.'