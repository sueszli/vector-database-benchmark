from dataclasses import dataclass
from typing import List
from UM import i18nCatalog
catalog = i18nCatalog('cura')

@dataclass
class ActiveQuality:
    """ Represents the active intent+profile combination, contains all information needed to display active quality. """
    intent_category: str = ''
    intent_name: str = ''
    profile: str = ''
    custom_profile: str = ''
    layer_height: float = None
    is_experimental: bool = False

    def getMainStringParts(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        string_parts = []
        if self.custom_profile is not None:
            string_parts.append(self.custom_profile)
        else:
            string_parts.append(self.profile)
            if self.intent_category != 'default':
                string_parts.append(self.intent_name)
        return string_parts

    def getTailStringParts(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        string_parts = []
        if self.custom_profile is not None:
            string_parts.append(self.profile)
            if self.intent_category != 'default':
                string_parts.append(self.intent_name)
        if self.layer_height:
            string_parts.append(f'{self.layer_height}mm')
        if self.is_experimental:
            string_parts.append(catalog.i18nc('@label', 'Experimental'))
        return string_parts

    def getStringParts(self) -> List[str]:
        if False:
            print('Hello World!')
        return self.getMainStringParts() + self.getTailStringParts()