"""Class that maps (chain_id, residue_id) to a residue property."""

class AbstractPropertyMap:
    """Define base class, map holder of residue properties."""

    def __init__(self, property_dict, property_keys, property_list):
        if False:
            return 10
        'Initialize the class.'
        self.property_dict = property_dict
        self.property_keys = property_keys
        self.property_list = property_list

    def _translate_id(self, entity_id):
        if False:
            while True:
                i = 10
        'Return entity identifier (PRIVATE).'
        return entity_id

    def __contains__(self, id):
        if False:
            return 10
        'Check if the mapping has a property for this residue.\n\n        :param chain_id: chain id\n        :type chain_id: char\n\n        :param res_id: residue id\n        :type res_id: char\n\n        Examples\n        --------\n        This is an incomplete but illustrative example::\n\n            if (chain_id, res_id) in apmap:\n                res, prop = apmap[(chain_id, res_id)]\n\n        '
        translated_id = self._translate_id(id)
        return translated_id in self.property_dict

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Return property for a residue.\n\n        :param chain_id: chain id\n        :type chain_id: char\n\n        :param res_id: residue id\n        :type res_id: int or (char, int, char)\n\n        :return: some residue property\n        :rtype: anything (can be a tuple)\n        '
        translated_id = self._translate_id(key)
        return self.property_dict[translated_id]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return number of residues for which the property is available.\n\n        :return: number of residues\n        :rtype: int\n        '
        return len(self.property_dict)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the list of residues.\n\n        :return: list of residues for which the property was calculated\n        :rtype: [(chain_id, res_id), (chain_id, res_id),...]\n        '
        return self.property_keys

    def __iter__(self):
        if False:
            return 10
        "Iterate over the (entity, property) list.\n\n        Handy alternative to the dictionary-like access.\n\n        :return: iterator\n\n        Examples\n        --------\n        >>> entity_property_list = [\n        ...     ('entity_1', 'property_1'),\n        ...     ('entity_2', 'property_2')\n        ... ]\n        >>> map = AbstractPropertyMap({}, [], entity_property_list)\n        >>> for (res, property) in iter(map):\n        ...     print(res, property)\n        entity_1 property_1\n        entity_2 property_2\n\n        "
        for i in range(len(self.property_list)):
            yield self.property_list[i]

class AbstractResiduePropertyMap(AbstractPropertyMap):
    """Define class for residue properties map."""

    def __init__(self, property_dict, property_keys, property_list):
        if False:
            print('Hello World!')
        'Initialize the class.'
        AbstractPropertyMap.__init__(self, property_dict, property_keys, property_list)

    def _translate_id(self, ent_id):
        if False:
            print('Hello World!')
        'Return entity identifier on residue (PRIVATE).'
        (chain_id, res_id) = ent_id
        if isinstance(res_id, int):
            ent_id = (chain_id, (' ', res_id, ' '))
        return ent_id

class AbstractAtomPropertyMap(AbstractPropertyMap):
    """Define class for atom properties map."""

    def __init__(self, property_dict, property_keys, property_list):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        AbstractPropertyMap.__init__(self, property_dict, property_keys, property_list)

    def _translate_id(self, ent_id):
        if False:
            return 10
        'Return entity identifier on atoms (PRIVATE).'
        if len(ent_id) == 4:
            (chain_id, res_id, atom_name, icode) = ent_id
        else:
            (chain_id, res_id, atom_name) = ent_id
            icode = None
        if isinstance(res_id, int):
            ent_id = (chain_id, (' ', res_id, ' '), atom_name, icode)
        return ent_id