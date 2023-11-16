"""Validation mixin class definition"""
import part.models
import stock.models

class ValidationMixin:
    """Mixin class that allows custom validation for various parts of InvenTree

    Custom generation and validation functionality can be provided for:

    - Part names
    - Part IPN (internal part number) values
    - Part parameter values
    - Serial numbers
    - Batch codes

    Notes:
    - Multiple ValidationMixin plugins can be used simultaneously
    - The stub methods provided here generally return None (null value).
    - The "first" plugin to return a non-null value for a particular method "wins"
    - In the case of "validation" functions, all loaded plugins are checked until an exception is thrown

    Implementing plugins may override any of the following methods which are of interest.

    For 'validation' methods, there are three 'acceptable' outcomes:
    - The method determines that the value is 'invalid' and raises a django.core.exceptions.ValidationError
    - The method passes and returns None (the code then moves on to the next plugin)
    - The method passes and returns True (and no subsequent plugins are checked)

    """

    class MixinMeta:
        """Metaclass for this mixin"""
        MIXIN_NAME = 'Validation'

    def __init__(self):
        if False:
            return 10
        'Register the mixin'
        super().__init__()
        self.add_mixin('validation', True, __class__)

    def validate_part_name(self, name: str, part: part.models.Part):
        if False:
            return 10
        'Perform validation on a proposed Part name\n\n        Arguments:\n            name: The proposed part name\n            part: The part instance we are validating against\n\n        Returns:\n            None or True (refer to class docstring)\n\n        Raises:\n            ValidationError if the proposed name is objectionable\n        '
        return None

    def validate_part_ipn(self, ipn: str, part: part.models.Part):
        if False:
            return 10
        'Perform validation on a proposed Part IPN (internal part number)\n\n        Arguments:\n            ipn: The proposed part IPN\n            part: The Part instance we are validating against\n\n        Returns:\n            None or True (refer to class docstring)\n\n        Raises:\n            ValidationError if the proposed IPN is objectionable\n        '
        return None

    def validate_batch_code(self, batch_code: str, item: stock.models.StockItem):
        if False:
            while True:
                i = 10
        'Validate the supplied batch code\n\n        Arguments:\n            batch_code: The proposed batch code (string)\n            item: The StockItem instance we are validating against\n\n        Returns:\n            None or True (refer to class docstring)\n\n        Raises:\n            ValidationError if the proposed batch code is objectionable\n        '
        return None

    def generate_batch_code(self):
        if False:
            return 10
        'Generate a new batch code\n\n        Returns:\n            A new batch code (string) or None\n        '
        return None

    def validate_serial_number(self, serial: str, part: part.models.Part):
        if False:
            while True:
                i = 10
        'Validate the supplied serial number.\n\n        Arguments:\n            serial: The proposed serial number (string)\n            part: The Part instance for which this serial number is being validated\n\n        Returns:\n            None or True (refer to class docstring)\n\n        Raises:\n            ValidationError if the proposed serial is objectionable\n        '
        return None

    def convert_serial_to_int(self, serial: str):
        if False:
            i = 10
            return i + 15
        'Convert a serial number (string) into an integer representation.\n\n        This integer value is used for efficient sorting based on serial numbers.\n\n        A plugin which implements this method can either return:\n\n        - An integer based on the serial string, according to some algorithm\n        - A fixed value, such that serial number sorting reverts to the string representation\n        - None (null value) to let any other plugins perform the conversion\n\n        Note that there is no requirement for the returned integer value to be unique.\n\n        Arguments:\n            serial: Serial value (string)\n\n        Returns:\n            integer representation of the serial number, or None\n        '
        return None

    def increment_serial_number(self, serial: str):
        if False:
            while True:
                i = 10
        'Return the next sequential serial based on the provided value.\n\n        A plugin which implements this method can either return:\n\n        - A string which represents the "next" serial number in the sequence\n        - None (null value) if the next value could not be determined\n\n        Arguments:\n            serial: Current serial value (string)\n        '
        return None

    def validate_part_parameter(self, parameter, data):
        if False:
            return 10
        'Validate a parameter value.\n\n        Arguments:\n            parameter: The parameter we are validating\n            data: The proposed parameter value\n\n        Returns:\n            None or True (refer to class docstring)\n\n        Raises:\n            ValidationError if the proposed parameter value is objectionable\n        '
        pass