import array
from six import string_types

class IP6_Address:
    ADDRESS_BYTE_SIZE = 16
    TOTAL_HEX_GROUPS = 8
    HEX_GROUP_SIZE = 4
    TOTAL_SEPARATORS = TOTAL_HEX_GROUPS - 1
    ADDRESS_TEXT_SIZE = TOTAL_HEX_GROUPS * HEX_GROUP_SIZE + TOTAL_SEPARATORS
    SEPARATOR = ':'
    SCOPE_SEPARATOR = '%'

    def __init__(self, address):
        if False:
            return 10
        self.__bytes = array.array('B', b'\x00' * self.ADDRESS_BYTE_SIZE)
        self.__scope_id = ''
        if isinstance(address, string_types):
            self.__from_string(address)
        else:
            self.__from_bytes(address)

    def __from_string(self, address):
        if False:
            for i in range(10):
                print('nop')
        if self.__is_a_scoped_address(address):
            split_parts = address.split(self.SCOPE_SEPARATOR)
            address = split_parts[0]
            if split_parts[1] == '':
                raise Exception('Empty scope ID')
            self.__scope_id = split_parts[1]
        if self.__is_address_in_compressed_form(address):
            address = self.__expand_compressed_address(address)
        address = self.__insert_leading_zeroes(address)
        if len(address) != self.ADDRESS_TEXT_SIZE:
            raise Exception('IP6_Address - from_string - address size != ' + str(self.ADDRESS_TEXT_SIZE))
        hex_groups = address.split(self.SEPARATOR)
        if len(hex_groups) != self.TOTAL_HEX_GROUPS:
            raise Exception('IP6_Address - parsed hex groups != ' + str(self.TOTAL_HEX_GROUPS))
        offset = 0
        for group in hex_groups:
            if len(group) != self.HEX_GROUP_SIZE:
                raise Exception('IP6_Address - parsed hex group length != ' + str(self.HEX_GROUP_SIZE))
            group_as_int = int(group, 16)
            self.__bytes[offset] = (group_as_int & 65280) >> 8
            self.__bytes[offset + 1] = group_as_int & 255
            offset += 2

    def __from_bytes(self, theBytes):
        if False:
            return 10
        if len(theBytes) != self.ADDRESS_BYTE_SIZE:
            raise Exception('IP6_Address - from_bytes - array size != ' + str(self.ADDRESS_BYTE_SIZE))
        self.__bytes = theBytes

    def as_string(self, compress_address=True, scoped_address=True):
        if False:
            return 10
        s = ''
        for (i, v) in enumerate(self.__bytes):
            s += hex(v)[2:].rjust(2, '0')
            if i % 2 == 1:
                s += self.SEPARATOR
        s = s[:-1].upper()
        if compress_address:
            s = self.__trim_leading_zeroes(s)
            s = self.__trim_longest_zero_chain(s)
        if scoped_address and self.get_scope_id() != '':
            s += self.SCOPE_SEPARATOR + self.__scope_id
        return s

    def as_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__bytes

    def __str__(self):
        if False:
            print('Hello World!')
        return self.as_string()

    def get_scope_id(self):
        if False:
            i = 10
            return i + 15
        return self.__scope_id

    def get_unscoped_address(self):
        if False:
            for i in range(10):
                print('nop')
        return self.as_string(True, False)

    def is_multicast(self):
        if False:
            i = 10
            return i + 15
        return self.__bytes[0] == 255

    def is_unicast(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__bytes[0] == 254

    def is_link_local_unicast(self):
        if False:
            i = 10
            return i + 15
        return self.is_unicast() and self.__bytes[1] & 192 == 128

    def is_site_local_unicast(self):
        if False:
            while True:
                i = 10
        return self.is_unicast() and self.__bytes[1] & 192 == 192

    def is_unique_local_unicast(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__bytes[0] == 253

    def get_human_readable_address_type(self):
        if False:
            print('Hello World!')
        if self.is_multicast():
            return 'multicast'
        elif self.is_unicast():
            if self.is_link_local_unicast():
                return 'link-local unicast'
            elif self.is_site_local_unicast():
                return 'site-local unicast'
            else:
                return 'unicast'
        elif self.is_unique_local_unicast():
            return 'unique-local unicast'
        else:
            return 'unknown type'

    def __is_address_in_compressed_form(self, address):
        if False:
            for i in range(10):
                print('nop')
        if address.count(self.SEPARATOR * 3) > 0:
            raise Exception('IP6_Address - found triple colon')
        compression_marker_count = self.__count_compression_marker(address)
        if compression_marker_count == 0:
            return False
        elif compression_marker_count == 1:
            return True
        else:
            raise Exception('IP6_Address - more than one compression marker ("::") found')

    def __count_compressed_groups(self, address):
        if False:
            i = 10
            return i + 15
        trimmed_address = address.replace(self.SEPARATOR * 2, self.SEPARATOR)
        return trimmed_address.count(self.SEPARATOR) + 1

    def __count_compression_marker(self, address):
        if False:
            print('Hello World!')
        return address.count(self.SEPARATOR * 2)

    def __insert_leading_zeroes(self, address):
        if False:
            print('Hello World!')
        hex_groups = address.split(self.SEPARATOR)
        new_address = ''
        for hex_group in hex_groups:
            if len(hex_group) < 4:
                hex_group = hex_group.rjust(4, '0')
            new_address += hex_group + self.SEPARATOR
        return new_address[:-1]

    def __expand_compressed_address(self, address):
        if False:
            while True:
                i = 10
        group_count = self.__count_compressed_groups(address)
        groups_to_insert = self.TOTAL_HEX_GROUPS - group_count
        pos = address.find(self.SEPARATOR * 2) + 1
        while groups_to_insert:
            address = address[:pos] + '0000' + self.SEPARATOR + address[pos:]
            pos += 5
            groups_to_insert -= 1
        address = address.replace(self.SEPARATOR * 2, self.SEPARATOR)
        return address

    def __trim_longest_zero_chain(self, address):
        if False:
            for i in range(10):
                print('nop')
        chain_size = 8
        while chain_size > 0:
            groups = address.split(self.SEPARATOR)
            for (index, group) in enumerate(groups):
                if group == '0':
                    start_index = index
                    end_index = index
                    while end_index < 7 and groups[end_index + 1] == '0':
                        end_index += 1
                    found_size = end_index - start_index + 1
                    if found_size == chain_size:
                        address = self.SEPARATOR.join(groups[0:start_index]) + self.SEPARATOR * 2 + self.SEPARATOR.join(groups[end_index + 1:])
                        return address
            chain_size -= 1
        return address

    def __trim_leading_zeroes(self, theStr):
        if False:
            i = 10
            return i + 15
        groups = theStr.split(self.SEPARATOR)
        theStr = ''
        for group in groups:
            group = group.lstrip('0') + self.SEPARATOR
            if group == self.SEPARATOR:
                group = '0' + self.SEPARATOR
            theStr += group
        return theStr[:-1]

    @classmethod
    def is_a_valid_text_representation(cls, text_representation):
        if False:
            i = 10
            return i + 15
        try:
            IP6_Address(text_representation)
            return True
        except Exception:
            return False

    def __is_a_scoped_address(self, text_representation):
        if False:
            for i in range(10):
                print('nop')
        return text_representation.count(self.SCOPE_SEPARATOR) == 1