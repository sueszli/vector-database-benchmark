import django.views.decorators.csrf.csrf_exempt
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def foo():
    if False:
        while True:
            i = 10
    return 1

@django.views.decorators.csrf.csrf_exempt
def foo():
    if False:
        print('Hello World!')
    return 1