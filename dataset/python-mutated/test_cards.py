from .card import MetaflowCard, MetaflowCardComponent

class TestStringComponent(MetaflowCardComponent):

    def __init__(self, text):
        if False:
            for i in range(10):
                print('nop')
        self._text = text

    def render(self):
        if False:
            i = 10
            return i + 15
        return str(self._text)

class TestPathSpecCard(MetaflowCard):
    type = 'test_pathspec_card'

    def render(self, task):
        if False:
            while True:
                i = 10
        import random
        import string
        return '%s %s' % (task.pathspec, ''.join((random.choice(string.ascii_uppercase + string.digits) for _ in range(6))))

class TestEditableCard(MetaflowCard):
    type = 'test_editable_card'
    seperator = '$&#!!@*'
    ALLOW_USER_COMPONENTS = True

    def __init__(self, options={}, components=[], graph=None):
        if False:
            print('Hello World!')
        self._components = components

    def render(self, task):
        if False:
            return 10
        return self.seperator.join([str(comp) for comp in self._components])

class TestEditableCard2(MetaflowCard):
    type = 'test_editable_card_2'
    seperator = '$&#!!@*'
    ALLOW_USER_COMPONENTS = True

    def __init__(self, options={}, components=[], graph=None):
        if False:
            for i in range(10):
                print('nop')
        self._components = components

    def render(self, task):
        if False:
            print('Hello World!')
        return self.seperator.join([str(comp) for comp in self._components])

class TestNonEditableCard(MetaflowCard):
    type = 'test_non_editable_card'
    seperator = '$&#!!@*'

    def __init__(self, options={}, components=[], graph=None):
        if False:
            print('Hello World!')
        self._components = components

    def render(self, task):
        if False:
            return 10
        return self.seperator.join([str(comp) for comp in self._components])

class TestMockCard(MetaflowCard):
    type = 'test_mock_card'

    def __init__(self, options={'key': 'dummy_key'}, **kwargs):
        if False:
            while True:
                i = 10
        self._key = options['key']

    def render(self, task):
        if False:
            return 10
        task_data = task[self._key].data
        return '%s' % task_data

class TestErrorCard(MetaflowCard):
    type = 'test_error_card'

    def render(self, task):
        if False:
            i = 10
            return i + 15
        raise Exception('Unknown Things Happened')

class TestTimeoutCard(MetaflowCard):
    type = 'test_timeout_card'

    def __init__(self, options={'timeout': 50}, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._timeout = 10
        if 'timeout' in options:
            self._timeout = options['timeout']

    def render(self, task):
        if False:
            while True:
                i = 10
        import time
        time.sleep(self._timeout)
        return '%s' % task.pathspec