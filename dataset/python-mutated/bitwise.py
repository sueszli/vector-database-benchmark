from ...core.smtlib import Operators
from ...core.smtlib.expression import BitVec

def Mask(width):
    if False:
        return 10
    '\n    Return a mask with the low `width` bits set.\n\n    :param int width: How many bits to set to 1\n    :return: int or long\n    '
    return (1 << width) - 1

def Bit(value, idx):
    if False:
        return 10
    '\n    Extract `idx` bit from `value`.\n\n    :param value: Source value from which to extract.\n    :type value: int or long or BitVec\n    :param idx: Bit index\n    :return: int or long or BitVex\n    '
    return Operators.EXTRACT(value, idx, 1)

def GetNBits(value, nbits):
    if False:
        return 10
    '\n    Get the first `nbits` from `value`.\n\n    :param value: Source value from which to extract\n    :type value: int or long or BitVec\n    :param int nbits: How many bits to extract\n    :return: Low `nbits` bits of `value`.\n    :rtype int or long or BitVec\n    '
    if isinstance(value, int):
        return Operators.EXTRACT(value, 0, nbits)
    elif isinstance(value, BitVec):
        if value.size < nbits:
            return Operators.ZEXTEND(value, nbits)
        else:
            return Operators.EXTRACT(value, 0, nbits)

def SInt(value, width):
    if False:
        i = 10
        return i + 15
    '\n    Convert a bitstring `value` of `width` bits to a signed integer\n    representation.\n\n    :param value: The value to convert.\n    :type value: int or long or BitVec\n    :param int width: The width of the bitstring to consider\n    :return: The converted value\n    :rtype int or long or BitVec\n    '
    return Operators.ITEBV(width, Bit(value, width - 1) == 1, GetNBits(value, width) - 2 ** width, GetNBits(value, width))

def UInt(value, width):
    if False:
        i = 10
        return i + 15
    '\n    Return integer value of `value` as a bitstring of `width` width.\n\n    :param value: The value to convert.\n    :type value: int or long or BitVec\n    :param int width: The width of the bitstring to consider\n    :return: The integer value\n    :rtype int or long or BitVec\n    '
    return GetNBits(value, width)

def LSL_C(value, amount, width, with_carry=True):
    if False:
        i = 10
        return i + 15
    '\n    The ARM LSL_C (logical left shift with carry) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value and the carry result\n    :rtype tuple\n    '
    if isinstance(amount, int):
        assert amount > 0
    value = Operators.ZEXTEND(value, width * 2)
    amount = Operators.ZEXTEND(amount, width * 2)
    shifted = value << amount
    result = GetNBits(shifted, width)
    if with_carry:
        carry = Bit(shifted, width)
        return (result, carry)
    else:
        return result

def LSL(value, amount, width):
    if False:
        i = 10
        return i + 15
    '\n    The ARM LSL (logical left shift) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value\n    :rtype int or BitVec\n    '
    if isinstance(amount, int) and amount == 0:
        return value
    result = LSL_C(value, amount, width, with_carry=False)
    return result

def LSR_C(value, amount, width, with_carry=True):
    if False:
        i = 10
        return i + 15
    '\n    The ARM LSR_C (logical shift right with carry) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value and carry result\n    :rtype tuple\n    '
    if isinstance(amount, int):
        assert amount > 0
    result = GetNBits(value >> amount, width)
    if with_carry:
        carry = Bit(value >> amount - 1, 0)
        return (result, carry)
    else:
        return result

def LSR(value, amount, width):
    if False:
        i = 10
        return i + 15
    '\n    The ARM LSR (logical shift right) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value\n    :rtype int or BitVec\n    '
    if isinstance(amount, int) and amount == 0:
        return value
    result = LSR_C(value, amount, width, with_carry=False)
    return result

def ASR_C(value, amount, width, with_carry=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    The ARM ASR_C (arithmetic shift right with carry) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value and carry result\n    :rtype tuple\n    '
    if isinstance(amount, int) and isinstance(width, int):
        assert amount <= width
    if isinstance(amount, int):
        assert amount > 0
    if isinstance(amount, int) and isinstance(width, int):
        assert amount + width <= width * 2
    value = Operators.SEXTEND(value, width, width * 2)
    amount = Operators.ZEXTEND(amount, width * 2)
    result = GetNBits(value >> amount, width)
    if with_carry:
        carry = Bit(value, amount - 1)
        return (result, carry)
    else:
        return result

def ASR(value, amount, width):
    if False:
        return 10
    '\n    The ARM ASR (arithmetic shift right) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to shift it.\n    :param int width: Width of the value\n    :return: Resultant value\n    :rtype int or BitVec\n    '
    if isinstance(amount, int) and amount == 0:
        return value
    result = ASR_C(value, amount, width, with_carry=False)
    return result

def ROR_C(value, amount, width, with_carry=True):
    if False:
        return 10
    '\n    The ARM ROR_C (rotate right with carry) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to rotate it.\n    :param int width: Width of the value\n    :return: Resultant value and carry result\n    :rtype tuple\n    '
    if isinstance(amount, int) and isinstance(width, int):
        assert amount <= width
    if isinstance(amount, int):
        assert amount > 0
    m = amount % width
    (right, _) = LSR_C(value, m, width)
    (left, _) = LSL_C(value, width - m, width)
    result = left | right
    if with_carry:
        carry = Bit(result, width - 1)
        return (result, carry)
    else:
        return result

def ROR(value, amount, width):
    if False:
        return 10
    '\n    The ARM ROR (rotate right) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to rotate it.\n    :param int width: Width of the value\n    :return: Resultant value\n    :rtype int or BitVec\n    '
    if isinstance(amount, int) and amount == 0:
        return value
    result = ROR_C(value, amount, width, with_carry=False)
    return result

def RRX_C(value, carry, width, with_carry=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    The ARM RRX (rotate right with extend and with carry) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to rotate it.\n    :param int width: Width of the value\n    :return: Resultant value and carry result\n    :rtype tuple\n    '
    result = value >> 1 | carry << width - 1
    if with_carry:
        carry_out = Bit(value, 0)
        return (result, carry_out)
    else:
        return result

def RRX(value, carry, width):
    if False:
        i = 10
        return i + 15
    '\n    The ARM RRX (rotate right with extend) operation.\n\n    :param value: Value to shift\n    :type value: int or long or BitVec\n    :param int amount: How many bits to rotate it.\n    :param int width: Width of the value\n    :return: Resultant value\n    :rtype int or BitVec\n    '
    result = RRX_C(value, carry, width, with_carry=False)
    return result