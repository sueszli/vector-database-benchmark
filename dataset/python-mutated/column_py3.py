from msrest.serialization import Model

class Column(Model):
    """A table column.

    A column in a table.

    :param name: The name of this column.
    :type name: str
    :param type: The data type of this column.
    :type type: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}, 'type': {'key': 'type', 'type': 'str'}}

    def __init__(self, *, name: str=None, type: str=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(Column, self).__init__(**kwargs)
        self.name = name
        self.type = type