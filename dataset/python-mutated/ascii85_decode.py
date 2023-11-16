"""
Decodes data encoded in an ASCII base-85 representation,
reproducing the original binary data.
"""
import base64

class ASCII85Decode:
    """
    Decodes data encoded in an ASCII base-85 representation,
    reproducing the original binary data.
    """

    @staticmethod
    def decode(bytes_in: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Decodes data encoded in an ASCII base-85 representation\n        '
        exceptions_to_throw = []
        if len(bytes_in) == 0:
            return bytes_in
        if bytes_in[-1] == 10 and bytes_in[-2] == 13:
            bytes_in = bytes_in[0:-2]
        if bytes_in[-1] == 10:
            bytes_in = bytes_in[0:-1]
        if bytes_in[-1] == 13:
            bytes_in = bytes_in[0:-1]
        try:
            return base64.a85decode(bytes_in)
        except Exception as e:
            exceptions_to_throw.append(e)
            pass
        try:
            return base64.a85decode(bytes_in, adobe=True)
        except Exception as e:
            exceptions_to_throw.append(e)
            pass
        raise exceptions_to_throw[0]