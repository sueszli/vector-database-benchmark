from typing import Dict, List, Type
from label_types.models import LabelType
from projects.models import Project

class LabelTypes:

    def __init__(self, label_type_class: Type[LabelType]):
        if False:
            print('Hello World!')
        self.types: Dict[str, LabelType] = {}
        self.label_type_class = label_type_class

    def __contains__(self, text: str) -> bool:
        if False:
            i = 10
            return i + 15
        return text in self.types

    def __getitem__(self, text: str) -> LabelType:
        if False:
            for i in range(10):
                print('nop')
        return self.types[text]

    def save(self, label_types: List[LabelType]):
        if False:
            for i in range(10):
                print('nop')
        self.label_type_class.objects.bulk_create(label_types, ignore_conflicts=True)

    def update(self, project: Project):
        if False:
            print('Hello World!')
        types = self.label_type_class.objects.filter(project=project)
        self.types = {label_type.text: label_type for label_type in types}