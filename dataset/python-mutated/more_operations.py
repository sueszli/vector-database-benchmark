from django.db.migrations.operations.base import Operation

class TestOperation(Operation):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def deconstruct(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.__class__.__name__, [], {})

    @property
    def reversible(self):
        if False:
            print('Hello World!')
        return True

    def state_forwards(self, app_label, state):
        if False:
            for i in range(10):
                print('nop')
        pass

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        if False:
            while True:
                i = 10
        pass

    def state_backwards(self, app_label, state):
        if False:
            return 10
        pass

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if False:
            i = 10
            return i + 15
        pass