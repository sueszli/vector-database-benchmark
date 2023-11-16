class BambooHrError(Exception):
    message = ''

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(self.message)

class NullFieldsError(BambooHrError):
    message = 'Field `custom_reports_fields` cannot be empty if `custom_reports_include_default_fields` is false.'

class AvailableFieldsAccessDeniedError(BambooHrError):
    message = "You hasn't access to any report fields. Please check your access level."

class CustomFieldsAccessDeniedError(Exception):

    def __init__(self, denied_fields):
        if False:
            while True:
                i = 10
        self.message = f"Access to fields: {', '.join(denied_fields)} - denied. Please check your access level."
        super().__init__(self.message)