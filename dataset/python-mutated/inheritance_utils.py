def inject_function(target_cls, *injected_cls):
    if False:
        print('Hello World!')
    '\n    Inject function to base directly.\n\n    :param target_cls:      nano extended class, e.g. nano.tf.keras.Model\n    :param injected_cls:    class with extended method for tf base\n    '
    for cls in injected_cls:
        for name in dir(cls):
            if not name.startswith('_'):
                if name in dir(target_cls):
                    old_f = getattr(target_cls, name)
                    setattr(target_cls, name + '_old', old_f)
                    setattr(target_cls, name, getattr(cls, name))
                else:
                    setattr(target_cls, name, getattr(cls, name))