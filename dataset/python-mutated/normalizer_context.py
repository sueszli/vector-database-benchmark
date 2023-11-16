from caffe2.python import context
from caffe2.python.modifier_context import ModifierContext, UseModifierBase

class NormalizerContext(ModifierContext, context.DefaultManaged):
    """
    provide context to allow param_info to have different normalizers
    """

    def has_normalizer(self, name):
        if False:
            print('Hello World!')
        return self._has_modifier(name)

    def get_normalizer(self, name):
        if False:
            return 10
        assert self.has_normalizer(name), '{} normalizer is not provided!'.format(name)
        return self._get_modifier(name)

class UseNormalizer(UseModifierBase):
    """
    context class to allow setting the current context.
    Example usage with layer:
        normalizers = {'norm1': norm1, 'norm2': norm2}
        with UseNormalizer(normalizers):
            norm = NormalizerContext.current().get_normalizer('norm1')
            layer(norm=norm)
    """

    def _context_class(self):
        if False:
            while True:
                i = 10
        return NormalizerContext