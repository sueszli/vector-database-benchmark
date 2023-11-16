"""
Contains structures and API-like objects for effects from AoC.
"""
from __future__ import annotations
import typing
from ..converter_object import ConverterObject
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer
    from openage.convert.value_object.read.value_members import ValueMember

class GenieEffectObject(ConverterObject):
    """
    Single effect contained in GenieEffectBundle.
    """
    __slots__ = ('bundle_id', 'data')

    def __init__(self, effect_id: int, bundle_id: int, full_data_set: GenieObjectContainer, members: dict[str, ValueMember]=None):
        if False:
            i = 10
            return i + 15
        "\n        Creates a new Genie effect object.\n\n        :param effect_id: The index of the effect in the .dat file's effect\n        :param bundle_id: The index of the effect bundle that the effect belongs to.\n                          (the index is referenced as tech_effect_id by techs)\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :param members: An already existing member dict.\n        "
        super().__init__(effect_id, members=members)
        self.bundle_id = bundle_id
        self.data = full_data_set

    def get_type(self) -> int:
        if False:
            return 10
        "\n        Returns the effect's type.\n        "
        return self['type_id'].value

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'GenieEffectObject<{self.get_id()}>'

class GenieEffectBundle(ConverterObject):
    """
    A set of effects of a tech.
    """
    __slots__ = ('effects', 'sanitized', 'data')

    def __init__(self, bundle_id: int, effects: list[GenieEffectObject], full_data_set: GenieObjectContainer, members: dict[str, ValueMember]=None):
        if False:
            i = 10
            return i + 15
        "\n        Creates a new Genie effect bundle.\n\n        :param bundle_id: The index of the effect in the .dat file's effect\n                          block. (the index is referenced as tech_effect_id by techs)\n        :param effects: Effects of the bundle as list of GenieEffectObject.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :param members: An already existing member dict.\n        "
        super().__init__(bundle_id, members=members)
        self.effects = effects
        self.sanitized: bool = False
        self.data = full_data_set

    def get_effects(self, effect_type: int=None) -> list[GenieEffectObject]:
        if False:
            while True:
                i = 10
        '\n        Returns the effects in the bundle, optionally only effects with a specific\n        type.\n\n        :param effect_type: Type that the effects should have.\n        :type effect_type: int\n        :returns: List of matching effects.\n        :rtype: list\n        '
        if effect_type:
            matching_effects = []
            for effect in self.effects.values():
                if effect.get_type() == effect_type:
                    matching_effects.append(effect)
            return matching_effects
        return list(self.effects.values())

    def is_sanitized(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns whether the effect bundle has been sanitized.\n        '
        return self.sanitized

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'GenieEffectBundle<{self.get_id()}>'