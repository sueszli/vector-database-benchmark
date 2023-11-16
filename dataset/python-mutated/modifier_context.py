DEFAULT_MODIFIER = 'DEFAULT'

class ModifierContext:
    """
    provide context to allow param_info to have different modifiers
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._modifiers = {}
        self._modifiers_list = []

    def _rebuild_modifiers(self):
        if False:
            print('Hello World!')
        self._modifiers = {}
        for m in self._modifiers_list:
            self._modifiers.update(m)

    def _has_modifier(self, name):
        if False:
            i = 10
            return i + 15
        return name in self._modifiers

    def _get_modifier(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._modifiers.get(name)

    def push_modifiers(self, modifiers):
        if False:
            return 10
        self._modifiers_list.append(modifiers)
        self._modifiers.update(modifiers)

    def pop_modifiers(self):
        if False:
            i = 10
            return i + 15
        assert len(self._modifiers_list) > 0
        self._modifiers_list.pop()
        self._rebuild_modifiers()

class UseModifierBase:
    """
    context class to allow setting the current context.
    Example usage with layer:
        modifiers = {'modifier1': modifier1, 'modifier2': modifier2}
        with Modifiers(modifiers):
            modifier = ModifierContext.current().get_modifier('modifier1')
            layer(modifier=modifier)
    """

    def __init__(self, modifier_or_dict):
        if False:
            return 10
        if isinstance(modifier_or_dict, dict):
            self._modifiers = modifier_or_dict
        else:
            self._modifiers = {DEFAULT_MODIFIER: modifier_or_dict}

    def _context_class(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._context_class().current().push_modifiers(self._modifiers)
        return self

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self._context_class().current().pop_modifiers()