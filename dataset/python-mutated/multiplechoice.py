from rest_framework import serializers
ERROR_MESSAGES = {'invalid_choice': 'Select a valid choice. {value} is not one of the available choices.'}

class MultipleChoiceField(serializers.Field):

    def __init__(self, choices=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.choices = set(choices or ())
        super().__init__(*args, **kwargs)

    def to_representation(self, value):
        if False:
            i = 10
            return i + 15
        return value

    def to_internal_value(self, data):
        if False:
            while True:
                i = 10
        if isinstance(data, list):
            for item in data:
                if item not in self.choices:
                    raise serializers.ValidationError(ERROR_MESSAGES['invalid_choice'].format(value=item))
            return data
        raise serializers.ValidationError('Please provide a valid list.')