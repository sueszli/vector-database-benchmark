from django import forms
from django.core.exceptions import ValidationError
from django.core.validators import validate_ipv4_address, validate_ipv6_address
from netaddr import IPAddress, IPNetwork, AddrFormatError

class IPAddressFormField(forms.Field):
    default_error_messages = {'invalid': 'Enter a valid IPv4 or IPv6 address (without a mask).'}

    def to_python(self, value):
        if False:
            print('Hello World!')
        if not value:
            return None
        if isinstance(value, IPAddress):
            return value
        try:
            validate_ipv4_address(value)
        except ValidationError:
            try:
                validate_ipv6_address(value)
            except ValidationError:
                raise ValidationError('Invalid IPv4/IPv6 address format: {}'.format(value))
        try:
            return IPAddress(value)
        except ValueError:
            raise ValidationError('This field requires an IP address without a mask.')
        except AddrFormatError:
            raise ValidationError('Please specify a valid IPv4 or IPv6 address.')

class IPNetworkFormField(forms.Field):
    default_error_messages = {'invalid': 'Enter a valid IPv4 or IPv6 address (with CIDR mask).'}

    def to_python(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not value:
            return None
        if isinstance(value, IPNetwork):
            return value
        if len(value.split('/')) != 2:
            raise ValidationError('CIDR mask (e.g. /24) is required.')
        try:
            return IPNetwork(value)
        except AddrFormatError:
            raise ValidationError('Please specify a valid IPv4 or IPv6 address.')