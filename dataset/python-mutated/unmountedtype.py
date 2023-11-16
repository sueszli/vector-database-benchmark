from ..utils.orderedtype import OrderedType

class UnmountedType(OrderedType):
    """
    This class acts a proxy for a Graphene Type, so it can be mounted
    dynamically as Field, InputField or Argument.

    Instead of writing:

    .. code:: python

        from graphene import ObjectType, Field, String

        class MyObjectType(ObjectType):
            my_field = Field(String, description='Description here')

    It lets you write:

    .. code:: python

        from graphene import ObjectType, String

        class MyObjectType(ObjectType):
            my_field = String(description='Description here')

    It is not used directly, but is inherited by other types and streamlines their use in
    different context:

    - Object Type
    - Scalar Type
    - Enum
    - Interface
    - Union

    An unmounted type will accept arguments based upon its context (ObjectType, Field or
    InputObjectType) and pass it on to the appropriate MountedType (Field, Argument or InputField).

    See each Mounted type reference for more information about valid parameters.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(UnmountedType, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def get_type(self):
        if False:
            while True:
                i = 10
        '\n        This function is called when the UnmountedType instance\n        is mounted (as a Field, InputField or Argument)\n        '
        raise NotImplementedError(f'get_type not implemented in {self}')

    def mount_as(self, _as):
        if False:
            for i in range(10):
                print('nop')
        return _as.mounted(self)

    def Field(self):
        if False:
            print('Hello World!')
        '\n        Mount the UnmountedType as Field\n        '
        from .field import Field
        return self.mount_as(Field)

    def InputField(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mount the UnmountedType as InputField\n        '
        from .inputfield import InputField
        return self.mount_as(InputField)

    def Argument(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mount the UnmountedType as Argument\n        '
        from .argument import Argument
        return self.mount_as(Argument)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self is other or (isinstance(other, UnmountedType) and self.get_type() == other.get_type() and (self.args == other.args) and (self.kwargs == other.kwargs))