from django.db import models
from django.db.models import Model

class TestModel1(models.Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        verbose_name = 'test model'
        verbose_name_plural = 'test models'

    @property
    def my_brand_new_property(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def my_beautiful_method(self):
        if False:
            i = 10
            return i + 15
        return 2

class TestModel2(Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        verbose_name = 'test model'
        verbose_name_plural = 'test models'

    @property
    def my_brand_new_property(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def my_beautiful_method(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

class TestModel3(Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = False

    @property
    def my_brand_new_property(self):
        if False:
            while True:
                i = 10
        return 1

    def my_beautiful_method(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

class TestModel4(Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        verbose_name = 'test model'
        verbose_name_plural = 'test models'

    def __str__(self):
        if False:
            return 10
        return self.new_field

    @property
    def my_brand_new_property(self):
        if False:
            print('Hello World!')
        return 1

    def my_beautiful_method(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

class TestModel5(models.Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        verbose_name = 'test model'
        verbose_name_plural = 'test models'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.new_field

    @property
    def my_brand_new_property(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def my_beautiful_method(self):
        if False:
            while True:
                i = 10
        return 2

class AbstractTestModel1(models.Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = True

    @property
    def my_brand_new_property(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def my_beautiful_method(self):
        if False:
            print('Hello World!')
        return 2

class AbstractTestModel2(Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = True

    @property
    def my_brand_new_property(self):
        if False:
            return 10
        return 1

    def my_beautiful_method(self):
        if False:
            while True:
                i = 10
        return 2

class AbstractTestModel3(Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = True

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.new_field

    @property
    def my_brand_new_property(self):
        if False:
            return 10
        return 1

    def my_beautiful_method(self):
        if False:
            return 10
        return 2

class AbstractTestModel4(models.Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = True

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.new_field

    @property
    def my_brand_new_property(self):
        if False:
            while True:
                i = 10
        return 1

    def my_beautiful_method(self):
        if False:
            print('Hello World!')
        return 2

class AbstractTestModel5(models.Model):
    new_field = models.CharField(max_length=10)

    class Meta:
        abstract = False

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.new_field

    @property
    def my_brand_new_property(self):
        if False:
            return 10
        return 1

    def my_beautiful_method(self):
        if False:
            i = 10
            return i + 15
        return 2