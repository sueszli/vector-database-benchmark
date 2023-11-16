from django.db.models.signals import pre_save
from django.dispatch import receiver
from myapp.models import MyModel
test_decorator = lambda func: lambda *args, **kwargs: func(*args, **kwargs)

@receiver(pre_save, sender=MyModel)
@test_decorator
def correct_pre_save_handler():
    if False:
        for i in range(10):
            print('nop')
    pass

@test_decorator
@receiver(pre_save, sender=MyModel)
def incorrect_pre_save_handler():
    if False:
        while True:
            i = 10
    pass

@receiver(pre_save, sender=MyModel)
@receiver(pre_save, sender=MyModel)
@test_decorator
def correct_multiple():
    if False:
        while True:
            i = 10
    pass

@receiver(pre_save, sender=MyModel)
@receiver(pre_save, sender=MyModel)
def correct_multiple():
    if False:
        while True:
            i = 10
    pass

@receiver(pre_save, sender=MyModel)
@test_decorator
@receiver(pre_save, sender=MyModel)
def incorrect_multiple():
    if False:
        i = 10
        return i + 15
    pass