"""factory_boy extensions for use with the mongoengine library (pymongo wrapper)."""
from . import base

class MongoEngineFactory(base.Factory):
    """Factory for mongoengine objects."""

    class Meta:
        abstract = True

    @classmethod
    def _build(cls, model_class, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return model_class(*args, **kwargs)

    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        if False:
            while True:
                i = 10
        instance = model_class(*args, **kwargs)
        if instance._is_document:
            instance.save()
        return instance