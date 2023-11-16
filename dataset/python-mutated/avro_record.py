"""AvroRecord for AvroGenericCoder."""
__all__ = ['AvroRecord']

class AvroRecord(object):
    """Simple wrapper class for dictionary records."""

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.record = value

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return issubclass(type(other), AvroRecord) and self.record == other.record

    def __hash__(self):
        if False:
            return 10
        return hash(self.record)