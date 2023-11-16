from django.utils.module_loading import autodiscover_modules
from django.utils.translation import gettext as _
from .wizard_base import Wizard

class AlreadyRegisteredException(Exception):
    pass

def entry_choices(user, page):
    if False:
        while True:
            i = 10
    '\n    Yields a list of wizard entries that the current user can use based on their\n    permission to add instances of the underlying model objects.\n    '
    for entry in wizard_pool.get_entries():
        if entry.user_has_add_permission(user, page=page):
            yield (entry.id, entry.title)

class WizardPool:
    _entries = {}
    _discovered = False

    def __init__(self):
        if False:
            while True:
                i = 10
        self._reset()

    def _discover(self):
        if False:
            while True:
                i = 10
        if not self._discovered:
            autodiscover_modules('cms_wizards')
            self._discovered = True

    def _clear(self):
        if False:
            while True:
                i = 10
        'Simply empties the pool but does not clear the discovered flag.'
        self._entries = {}

    def _reset(self):
        if False:
            while True:
                i = 10
        'Clears the wizard pool and clears the discovered flag.'
        self._clear()
        self._discovered = False

    @property
    def discovered(self):
        if False:
            while True:
                i = 10
        '\n        A public getter for the private property _discovered. Note, there is no\n        public setter.\n        '
        return self._discovered

    def is_registered(self, entry, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the provided entry is registered.\n\n        NOTE: This method triggers pool discovery unless a «passive» kwarg\n        is set to True\n        '
        passive = kwargs.get('passive', False)
        if not passive:
            self._discover()
        return entry.id in self._entries

    def register(self, entry):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers the provided «entry».\n\n        Raises AlreadyRegisteredException if the entry is already registered.\n        '
        assert isinstance(entry, Wizard), u'entry must be an instance of Wizard'
        if self.is_registered(entry, passive=True):
            model = entry.get_model()
            raise AlreadyRegisteredException(_(u'A wizard has already been registered for model: %s') % model.__name__)
        else:
            self._entries[entry.id] = entry

    def unregister(self, entry):
        if False:
            print('Hello World!')
        '\n        If «entry» is registered into the pool, remove it.\n\n        Returns True if the entry was successfully registered, else False.\n\n        NOTE: This method triggers pool discovery.\n        '
        assert isinstance(entry, Wizard), u'entry must be an instance of Wizard'
        if self.is_registered(entry, passive=True):
            del self._entries[entry.id]
            return True
        return False

    def get_entry(self, entry):
        if False:
            return 10
        '\n        Returns the wizard from the pool identified by «entry», which may be a\n        Wizard instance or its "id" (which is the PK of its underlying\n        content-type).\n\n        NOTE: This method triggers pool discovery.\n        '
        self._discover()
        if isinstance(entry, Wizard):
            entry = entry.id
        return self._entries[entry]

    def get_entries(self):
        if False:
            return 10
        '\n        Returns all entries in weight-order.\n\n        NOTE: This method triggers pool discovery.\n        '
        self._discover()
        return [value for (key, value) in sorted(self._entries.items(), key=lambda e: e[1].weight)]
wizard_pool = WizardPool()