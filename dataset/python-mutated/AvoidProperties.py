from functools import cached_property

class NonDataDescriptor:

    def __init__(self, func):
        if False:
            print('Hello World!')
        self.func = func

    def __get__(self, instance, owner):
        if False:
            for i in range(10):
                print('nop')
        return self.func(instance)

class DataDescriptor(NonDataDescriptor):

    def __set__(self, instance, value):
        if False:
            return 10
        pass

class FailingNonDataDescriptor(NonDataDescriptor):

    def __get__(self, instance, owner):
        if False:
            i = 10
            return i + 15
        return 1 / 0

class FailingDataDescriptor(DataDescriptor):

    def __get__(self, instance, owner):
        if False:
            print('Hello World!')
        return 1 / 0

class AvoidProperties:
    normal_property_called = 0
    classmethod_property_called = 0
    cached_property_called = 0
    non_data_descriptor_called = 0
    classmethod_non_data_descriptor_called = 0
    data_descriptor_called = 0
    classmethod_data_descriptor_called = 0

    def keyword(self):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def normal_property(self):
        if False:
            i = 10
            return i + 15
        type(self).normal_property_called += 1
        return self.normal_property_called

    @classmethod
    @property
    def classmethod_property(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.classmethod_property_called += 1
        return cls.classmethod_property_called

    @cached_property
    def cached_property(self):
        if False:
            while True:
                i = 10
        type(self).cached_property_called += 1
        return self.cached_property_called

    @NonDataDescriptor
    def non_data_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        type(self).non_data_descriptor_called += 1
        return self.non_data_descriptor_called

    @classmethod
    @NonDataDescriptor
    def classmethod_non_data_descriptor(cls):
        if False:
            return 10
        cls.classmethod_non_data_descriptor_called += 1
        return cls.classmethod_non_data_descriptor_called

    @DataDescriptor
    def data_descriptor(self):
        if False:
            print('Hello World!')
        type(self).data_descriptor_called += 1
        return self.data_descriptor_called

    @classmethod
    @DataDescriptor
    def classmethod_data_descriptor(cls):
        if False:
            i = 10
            return i + 15
        cls.classmethod_data_descriptor_called += 1
        return cls.classmethod_data_descriptor_called

    @FailingNonDataDescriptor
    def failing_non_data_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    @FailingNonDataDescriptor
    def failing_classmethod_non_data_descriptor(self):
        if False:
            i = 10
            return i + 15
        pass

    @FailingDataDescriptor
    def failing_data_descriptor(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    @FailingDataDescriptor
    def failing_classmethod_data_descriptor(self):
        if False:
            while True:
                i = 10
        pass