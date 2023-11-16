from abc import ABC, abstractmethod

class Macro(ABC):
    """https://docs.getdbt.com/docs/building-a-dbt-project/jinja-macros"""

    @abstractmethod
    def __str__(self):
        if False:
            return 10
        pass

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self)

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return str(self) + str(other)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return str(other) + str(self)

class Source(Macro):
    """https://docs.getdbt.com/reference/dbt-jinja-functions/source"""

    def __init__(self, source_name: str, table_name: str):
        if False:
            while True:
                i = 10
        self.source_name = source_name
        self.table_name = table_name

    def __str__(self):
        if False:
            print('Hello World!')
        return "source('{}', '{}')".format(self.source_name, self.table_name)

class Ref(Macro):
    """https://docs.getdbt.com/reference/dbt-jinja-functions/ref"""

    def __init__(self, model_name: str):
        if False:
            for i in range(10):
                print('nop')
        self.model_name = model_name

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return "ref('{}')".format(self.model_name)