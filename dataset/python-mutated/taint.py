from django.http import HttpRequest

def source() -> str:
    if False:
        i = 10
        return i + 15
    request = HttpRequest()
    return request.GET['bad']

def sink(argument: str) -> None:
    if False:
        return 10
    eval(argument)