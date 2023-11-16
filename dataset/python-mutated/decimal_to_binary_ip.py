"""
Given an ip address in dotted-decimal representation, determine the
binary representation. For example,
decimal_to_binary(255.0.0.5) returns 11111111.00000000.00000000.00000101
accepts string
returns string
"""

def decimal_to_binary_util(val):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert 8-bit decimal number to binary representation\n    :type val: str\n    :rtype: str\n    '
    bits = [128, 64, 32, 16, 8, 4, 2, 1]
    val = int(val)
    binary_rep = ''
    for bit in bits:
        if val >= bit:
            binary_rep += str(1)
            val -= bit
        else:
            binary_rep += str(0)
    return binary_rep

def decimal_to_binary_ip(ip):
    if False:
        print('Hello World!')
    '\n    Convert dotted-decimal ip address to binary representation with help of decimal_to_binary_util\n    '
    values = ip.split('.')
    binary_list = []
    for val in values:
        binary_list.append(decimal_to_binary_util(val))
    return '.'.join(binary_list)