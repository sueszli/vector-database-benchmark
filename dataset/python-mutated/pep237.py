from six.moves import builtins
original_hex = builtins.hex

def hex(number):
    if False:
        return 10
    original_hex.__doc__
    return original_hex(number).rstrip('L')
builtins.hex = hex