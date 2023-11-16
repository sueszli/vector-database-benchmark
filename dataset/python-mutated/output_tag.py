from typing import Optional, Union
from pyflink.common.typeinfo import TypeInformation, Types, RowTypeInfo
from pyflink.java_gateway import get_gateway

class OutputTag(object):
    """
    An :class:`OutputTag` is a typed and named tag to use for tagging side outputs of an operator.

    Example:
    ::

        # Explicitly specify output type
        >>> info = OutputTag("late-data", Types.TUPLE([Types.STRING(), Types.LONG()]))
        # Implicitly wrap list to Types.ROW
        >>> info_row = OutputTag("row", [Types.STRING(), Types.LONG()])
        # Implicitly use pickle serialization
        >>> info_side = OutputTag("side")
        # ERROR: tag id cannot be empty string (extra requirement for Python API)
        >>> info_error = OutputTag("")

    """

    def __init__(self, tag_id: str, type_info: Optional[Union[TypeInformation, list]]=None):
        if False:
            i = 10
            return i + 15
        if not tag_id:
            raise ValueError('OutputTag tag_id cannot be None or empty string')
        self.tag_id = tag_id
        if type_info is None:
            self.type_info = Types.PICKLED_BYTE_ARRAY()
        elif isinstance(type_info, list):
            self.type_info = RowTypeInfo(type_info)
        elif not isinstance(type_info, TypeInformation):
            raise TypeError('OutputTag type_info must be None, list or TypeInformation')
        else:
            self.type_info = type_info
        self._j_output_tag = None

    def __getstate__(self):
        if False:
            print('Hello World!')
        self.type_info._j_typeinfo = None
        return (self.tag_id, self.type_info)

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        (tag_id, type_info) = state
        self.tag_id = tag_id
        self.type_info = type_info
        self._j_output_tag = None

    def get_java_output_tag(self):
        if False:
            print('Hello World!')
        gateway = get_gateway()
        if self._j_output_tag is None:
            self._j_output_tag = gateway.jvm.org.apache.flink.util.OutputTag(self.tag_id, self.type_info.get_java_type_info())
        return self._j_output_tag