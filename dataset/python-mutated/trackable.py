import tensorflow as tf
from keras.utils import tracking

class KerasAutoTrackable(tf.__internal__.tracking.AutoTrackable):
    """Manages dependencies on other objects with Keras tracking.

    Similar to TF AutoTrackable, but disabling tracking is based
    on tracking within Keras.

    This serves as an interface between Keras tracking and TF tracking.
    """

    def __setattr__(self, name, value):
        if False:
            return 10
        'Support self.foo = trackable syntax.'
        try:
            if getattr(self, name) is value:
                return
        except AttributeError:
            pass
        if getattr(self, '_self_setattr_tracking', True):
            value = sticky_attribute_assignment(trackable=self, value=value, name=name)
        super().__setattr__(name, value)

def sticky_attribute_assignment(trackable, name, value):
    if False:
        return 10
    'Adds dependencies, called from __setattr__.\n\n    Args:\n        trackable: The object to add dependencies to (generally the one having\n        an attribute assigned).\n        name: The attribute name being assigned.\n        value: The value being assigned. Not necessarily a trackable object.\n\n    Returns:\n        The value which should be stored in the attribute.\n    '
    if isinstance(value, (tracking.TrackedList, tracking.TrackedDict, tracking.TrackedSet)) and hasattr(trackable, '_tracked'):
        trackable._tracked.append(name)
    if not tracking.is_tracking_enabled():
        return value
    if isinstance(value, tf.__internal__.tracking.Trackable):
        trackable._track_trackable(value, name=name, overwrite=True)
    return value