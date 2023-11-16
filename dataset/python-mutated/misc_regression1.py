import django.db.models
from django.db import models

class A(models.Model):

    def save(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self