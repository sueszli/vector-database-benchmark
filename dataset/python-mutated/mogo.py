"""factory_boy extensions for use with the mogo library (pymongo wrapper)."""
from . import base

class MogoFactory(base.Factory):
    """Factory for mogo objects."""

    class Meta:
        abstract = True

    @classmethod
    def _build(cls, model_class, *args, **kwargs):
        if False:
            print('Hello World!')
        return model_class(*args, **kwargs)

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        if False:
            return 10
        instance = model_class(*args, **kwargs)
        instance.save()
        return instance