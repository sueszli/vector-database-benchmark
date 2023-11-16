from django.utils.translation import gettext_lazy as _
from rest_framework import serializers
from netaddr import AddrFormatError, IPNetwork
__all__ = ('IPAddressField', 'IPNetworkField')

class IPAddressField(serializers.CharField):
    """
    An IPv4 or IPv6 address with optional mask
    """
    default_error_messages = {'invalid': _('Enter a valid IPv4 or IPv6 address with optional mask.')}

    def to_internal_value(self, data):
        if False:
            i = 10
            return i + 15
        try:
            return IPNetwork(data)
        except AddrFormatError:
            raise serializers.ValidationError(_('Invalid IP address format: {data}').format(data))
        except (TypeError, ValueError) as e:
            raise serializers.ValidationError(e)

    def to_representation(self, value):
        if False:
            while True:
                i = 10
        return str(value)

class IPNetworkField(serializers.CharField):
    """
    An IPv4 or IPv6 prefix
    """
    default_error_messages = {'invalid': _('Enter a valid IPv4 or IPv6 prefix and mask in CIDR notation.')}

    def to_internal_value(self, data):
        if False:
            i = 10
            return i + 15
        try:
            return IPNetwork(data)
        except AddrFormatError:
            raise serializers.ValidationError(_('Invalid IP prefix format: {data}').format(data))
        except (TypeError, ValueError) as e:
            raise serializers.ValidationError(e)

    def to_representation(self, value):
        if False:
            for i in range(10):
                print('nop')
        return str(value)