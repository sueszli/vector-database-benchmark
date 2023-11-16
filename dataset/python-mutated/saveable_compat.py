"""Checkpoint compatibility functions with SaveableObject.

Compatibility methods to ensure that checkpoints are saved with the same
metadata attributes before/after the SaveableObject deprecation.
"""
_LEGACY_SAVEABLE_NAME = '_LEGACY_SAVEABLE_NAME'

def legacy_saveable_name(name):
    if False:
        for i in range(10):
            print('nop')
    'Decorator to set the local name to use in the Checkpoint.\n\n  Needed for migrating certain Trackables (see next paragraph) from the legacy\n  `_gather_saveables_for_checkpoint` to the new `_serialize_to_tensors`\n  function.\n\n  This decorator should be used if the SaveableObject generates tensors with\n  different names from the name that is passed to the factory.\n\n  Example migration:\n\n  *Before*\n\n  ```\n  class MyTrackable(Trackable):\n    def _gather_saveables_for_checkpoint(self):\n      return {"key": _MySaveable}\n\n  class _MySaveable(SaveableObject):\n    def __init__(self, name):\n      specs = [\n          SaveSpec(tensor1, "", name + "-1")\n          SaveSpec(tensor2, "", name + "-2")\n      ]\n      super().__init__(None, specs, name)\n  ```\n\n  *After*\n\n  ```\n  @legacy_saveable_name("key")\n  class MyTrackable(Trackable):\n\n    def _serialize_to_tensors(self):\n      return {"key-1": tensor1, "key-2": tensor2}\n  ```\n\n  Args:\n    name: String name of the SaveableObject factory (the key returned in the\n       `_gather_saveables_for_checkpoint` function)\n\n  Returns:\n    A decorator.\n  '

    def decorator(cls_or_obj):
        if False:
            print('Hello World!')
        setattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, name)
        return cls_or_obj
    return decorator

def get_saveable_name(cls_or_obj):
    if False:
        print('Hello World!')
    return getattr(cls_or_obj, _LEGACY_SAVEABLE_NAME, None)
_FORCE_CHECKPOINT_CONVERSION = False

def force_checkpoint_conversion(value=True):
    if False:
        return 10
    'Forces checkpoint to use the new implementation.\n\n  The new checkpoint implementation is changing the saved metadata slightly,\n  and therefore may break forward compatibility in newly saved checkpoints. This\n  means:\n\n    - Previous versions of TensorFlow may not be able to load new checkpoints.\n    - Backwards compatibility is unchanged: Old checkpoints can still be loaded.\n\n  TensorFlow guarantees 3 weeks of forward compatibility, so this flag will be\n  removed in the future weeks, after which checkpoint conversion will happen by\n  default.\n\n  **What happens when this flag is enabled?**\n\n  The checkpoint will be saved with different metadata, meaning that previous\n  versions of TensorFlow (<=2.10) will not be able to load this checkpoint.\n\n  Args:\n    value: Boolean value, whether or not to force checkpoint conversion to the\n      new implementation.\n  '
    global _FORCE_CHECKPOINT_CONVERSION
    _FORCE_CHECKPOINT_CONVERSION = value

def force_checkpoint_conversion_enabled():
    if False:
        return 10
    return _FORCE_CHECKPOINT_CONVERSION

class CheckpointConversionError(Exception):
    pass