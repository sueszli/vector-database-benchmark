"""Sanitizer for body fields sent via Google Cloud API.

The sanitizer removes fields specified from the body.

Context
-------
In some cases where Google Cloud operation requires modification of existing resources (such
as instances or instance templates) we need to sanitize body of the resources returned
via Google Cloud APIs. This is in the case when we retrieve information from Google Cloud first,
modify the body and either update the existing resource or create a new one with the
modified body. Usually when you retrieve resource from Google Cloud you get some extra fields which
are Output-only, and we need to delete those fields if we want to use
the body as input for subsequent create/insert type operation.


Field specification
-------------------

Specification of fields is an array of strings which denote names of fields to be removed.
The field can be either direct field name to remove from the body or the full
specification of the path you should delete - separated with '.'


>>> FIELDS_TO_SANITIZE = [
>>>    "kind",
>>>    "properties.disks.kind",
>>>    "properties.metadata.kind",
>>>]
>>> body = {
>>>     "kind": "compute#instanceTemplate",
>>>     "name": "instance",
>>>     "properties": {
>>>         "disks": [
>>>             {
>>>                 "name": "a",
>>>                 "kind": "compute#attachedDisk",
>>>                 "type": "PERSISTENT",
>>>                 "mode": "READ_WRITE",
>>>             },
>>>             {
>>>                 "name": "b",
>>>                 "kind": "compute#attachedDisk",
>>>                 "type": "PERSISTENT",
>>>                 "mode": "READ_WRITE",
>>>             }
>>>         ],
>>>         "metadata": {
>>>             "kind": "compute#metadata",
>>>             "fingerprint": "GDPUYxlwHe4="
>>>         },
>>>     }
>>> }
>>> sanitizer=GcpBodyFieldSanitizer(FIELDS_TO_SANITIZE)
>>> sanitizer.sanitize(body)
>>> json.dumps(body, indent=2)
{
    "name":  "instance",
    "properties": {
        "disks": [
            {
                "name": "a",
                "type": "PERSISTENT",
                "mode": "READ_WRITE",
            },
            {
                "name": "b",
                "type": "PERSISTENT",
                "mode": "READ_WRITE",
            }
        ],
        "metadata": {
            "fingerprint": "GDPUYxlwHe4="
        },
    }
}

Note that the components of the path can be either dictionaries or arrays of dictionaries.
In case  they are dictionaries, subsequent component names key of the field, in case of
arrays - the sanitizer iterates through all dictionaries in the array and searches
components in all elements of the array.
"""
from __future__ import annotations
from airflow.exceptions import AirflowException
from airflow.utils.log.logging_mixin import LoggingMixin

class GcpFieldSanitizerException(AirflowException):
    """Thrown when sanitizer finds unexpected field type in the path (other than dict or array)."""

class GcpBodyFieldSanitizer(LoggingMixin):
    """Sanitizes the body according to specification.

    :param sanitize_specs: array of strings that specifies which fields to remove

    """

    def __init__(self, sanitize_specs: list[str]) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._sanitize_specs = sanitize_specs

    def _sanitize(self, dictionary, remaining_field_spec, current_path):
        if False:
            print('Hello World!')
        field_split = remaining_field_spec.split('.', 1)
        if len(field_split) == 1:
            field_name = field_split[0]
            if field_name in dictionary:
                self.log.info('Deleted %s [%s]', field_name, current_path)
                del dictionary[field_name]
            else:
                self.log.debug('The field %s is missing in %s at the path %s.', field_name, dictionary, current_path)
        else:
            field_name = field_split[0]
            remaining_path = field_split[1]
            child = dictionary.get(field_name)
            if child is None:
                self.log.debug('The field %s is missing in %s at the path %s. ', field_name, dictionary, current_path)
            elif isinstance(child, dict):
                self._sanitize(child, remaining_path, f'{current_path}.{field_name}')
            elif isinstance(child, list):
                for (index, elem) in enumerate(child):
                    if not isinstance(elem, dict):
                        self.log.warning('The field %s element at index %s is of wrong type. It should be dict and is %s. Skipping it.', current_path, index, elem)
                    self._sanitize(elem, remaining_path, f'{current_path}.{field_name}[{index}]')
            else:
                self.log.warning('The field %s is of wrong type. It should be dict or list and it is %s. Skipping it.', current_path, child)

    def sanitize(self, body) -> None:
        if False:
            print('Hello World!')
        'Sanitizes the body according to specification.'
        for elem in self._sanitize_specs:
            self._sanitize(body, elem, '')