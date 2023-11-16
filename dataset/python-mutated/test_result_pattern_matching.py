from returns.result import Failure, Success, safe

@safe
def div(first_number: int, second_number: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return first_number // second_number
match div(1, 0):
    case Success(10):
        print('Result is "10"')
    case Success(value):
        print('Result is "{0}"'.format(value))
    case Failure(ZeroDivisionError()):
        print('"ZeroDivisionError" was raised')
    case Failure(_):
        print('The division was a failure')