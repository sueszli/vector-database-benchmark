from django.db import connections, models
from django.db.models.signals import pre_migrate
from sentry.db.models.utils import Creator
__all__ = ('CITextField', 'CICharField', 'CIEmailField')

class CIText:

    def db_type(self, connection):
        if False:
            while True:
                i = 10
        return 'citext'

class CITextField(CIText, models.TextField):

    def contribute_to_class(self, cls, name):
        if False:
            while True:
                i = 10
        super().contribute_to_class(cls, name)
        setattr(cls, name, Creator(self))

class CICharField(CIText, models.CharField):

    def contribute_to_class(self, cls, name):
        if False:
            print('Hello World!')
        super().contribute_to_class(cls, name)
        setattr(cls, name, Creator(self))

class CIEmailField(CIText, models.EmailField):

    def contribute_to_class(self, cls, name):
        if False:
            i = 10
            return i + 15
        super().contribute_to_class(cls, name)
        setattr(cls, name, Creator(self))

def create_citext_extension(using, **kwargs):
    if False:
        return 10
    cursor = connections[using].cursor()
    try:
        cursor.execute('CREATE EXTENSION IF NOT EXISTS citext')
    except Exception:
        pass
pre_migrate.connect(create_citext_extension)