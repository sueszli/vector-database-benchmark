from typing import Iterator
from strawberry.extensions.base_extension import SchemaExtension

class DisableValidation(SchemaExtension):
    """
    Disable query validation

    Example:

    >>> import strawberry
    >>> from strawberry.extensions import DisableValidation
    >>>
    >>> schema = strawberry.Schema(
    ...     Query,
    ...     extensions=[
    ...         DisableValidation,
    ...     ]
    ... )

    """

    def __init__(self):
        if False:
            return 10
        pass

    def on_operation(self) -> Iterator[None]:
        if False:
            print('Hello World!')
        self.execution_context.validation_rules = ()
        yield