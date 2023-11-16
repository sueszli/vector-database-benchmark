from typing import Any, Callable, Dict, Optional

from .value_element import ValueElement


class ValidationElement(ValueElement):

    def __init__(self, validation: Dict[str, Callable[..., bool]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.validation = validation
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        """The latest error message from the validation functions."""
        return self._error

    def validate(self) -> None:
        """Validate the current value and set the error message if necessary."""
        for message, check in self.validation.items():
            if not check(self.value):
                self._error = message
                self.props(f'error error-message="{message}"')
                break
        else:
            self._error = None
            self.props(remove='error')

    def _handle_value_change(self, value: Any) -> None:
        super()._handle_value_change(value)
        self.validate()
