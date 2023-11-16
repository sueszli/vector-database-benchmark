from __future__ import annotations
import inspect
import itertools
from typing import Any, Callable, Iterable, Sequence, Tuple, Type
import pydantic
from django.utils.functional import LazyObject
from sentry.services.hybrid_cloud import ArgumentDict

class _SerializableFunctionSignatureException(Exception):

    def __init__(self, signature: SerializableFunctionSignature, message: str) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(f"{signature.generate_name('.')}: {message}")

class SerializableFunctionSignatureSetupException(_SerializableFunctionSignatureException):
    """Indicate that a function signature can't be set up for serialization."""

class SerializableFunctionValueException(_SerializableFunctionSignatureException):
    """Indicate that a serialized function call received an invalid value."""

class SerializableFunctionSignature:
    """Represent a function's parameters and return type for serialization."""

    def __init__(self, base_function: Callable[..., Any], is_instance_method: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.base_function = base_function
        self.is_instance_method = is_instance_method
        self._parameter_model = self._create_parameter_model()
        self._return_model = self._create_return_model()

    def get_name_segments(self) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        return (self.base_function.__name__,)

    def generate_name(self, joiner: str, suffix: str | None=None) -> str:
        if False:
            while True:
                i = 10
        segments: Iterable[str] = self.get_name_segments()
        if suffix is not None:
            segments = itertools.chain(segments, (suffix,))
        return joiner.join(segments)

    def _validate_type_token(self, value_label: str, token: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check whether a type token is usable.\n\n        Strings as type annotations, which Mypy can use if their types are imported\n        in an `if TYPE_CHECKING` block, can\'t be used for (de)serialization. Raise an\n        exception if the given token is one of these.\n\n        We can check only on a best-effort basis. String tokens may still be nested\n        in type parameters (e.g., `Optional["RpcThing"]`), which this won\'t catch.\n        Such a state would cause an exception when we attempt to use the signature\n        object to (de)serialize something.\n        '
        if isinstance(token, str):
            raise SerializableFunctionSignatureSetupException(self, f'Invalid type token on {value_label} (serializable functions must use concrete type tokens, not strings)')

    def _create_parameter_model(self) -> Type[pydantic.BaseModel]:
        if False:
            i = 10
            return i + 15
        'Dynamically create a Pydantic model class representing the parameters.'

        def create_field(param: inspect.Parameter) -> Tuple[Any, Any]:
            if False:
                for i in range(10):
                    print('nop')
            if param.annotation is param.empty:
                raise SerializableFunctionSignatureSetupException(self, 'Type annotations are required to serialize')
            self._validate_type_token(f'parameter `{param.name}`', param.annotation)
            default_value = ... if param.default is param.empty else param.default
            return (param.annotation, default_value)
        model_name = self.generate_name('__', 'ParameterModel')
        parameters = list(inspect.signature(self.base_function).parameters.values())
        if self.is_instance_method:
            parameters = parameters[1:]
        field_definitions = {p.name: create_field(p) for p in parameters}
        return pydantic.create_model(model_name, **field_definitions)
    _RETURN_MODEL_ATTR = 'value'

    def _create_return_model(self) -> Type[pydantic.BaseModel] | None:
        if False:
            for i in range(10):
                print('nop')
        "Dynamically create a Pydantic model class representing the return value.\n\n        The created model has a single attribute containing the return value. This\n        extra abstraction is necessary in order to have Pydantic handle generic\n        return annotations such as `Optional[RpcOrganization]` or `List[RpcUser]`,\n        where we can't directly access an RpcModel class on which to call `parse_obj`.\n        "
        model_name = self.generate_name('__', 'ReturnModel')
        return_type = inspect.signature(self.base_function).return_annotation
        if return_type is None:
            return None
        self._validate_type_token('return type', return_type)
        field_definitions = {self._RETURN_MODEL_ATTR: (return_type, ...)}
        return pydantic.create_model(model_name, **field_definitions)

    @staticmethod
    def _unwrap_lazy_django_object(arg: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "Unwrap any lazy objects before attempting to serialize.\n\n        It's possible to receive a SimpleLazyObject initialized by the Django\n        framework and pass it to an RPC (typically `request.user` as an RpcUser\n        argument). These objects are supposed to behave seamlessly like the\n        underlying type, but don't play nice with the reflection that Pydantic uses\n        to serialize. So, we manually check and force them to unwrap.\n        "
        if isinstance(arg, LazyObject):
            return getattr(arg, '_wrapped')
        else:
            return arg

    def serialize_arguments(self, raw_arguments: ArgumentDict) -> ArgumentDict:
        if False:
            for i in range(10):
                print('nop')
        raw_arguments = {key: self._unwrap_lazy_django_object(arg) for (key, arg) in raw_arguments.items()}
        try:
            model_instance = self._parameter_model(**raw_arguments)
        except Exception as e:
            raise SerializableFunctionValueException(self, 'Could not serialize arguments') from e
        return model_instance.dict()

    def deserialize_arguments(self, serial_arguments: ArgumentDict) -> pydantic.BaseModel:
        if False:
            return 10
        try:
            return self._parameter_model.parse_obj(serial_arguments)
        except Exception as e:
            raise SerializableFunctionValueException(self, 'Could not deserialize arguments') from e

    def deserialize_return_value(self, value: Any) -> Any:
        if False:
            print('Hello World!')
        if self._return_model is None:
            if value is not None:
                raise SerializableFunctionValueException(self, f'Expected None but got {type(value)}')
            return None
        parsed = self._return_model.parse_obj({self._RETURN_MODEL_ATTR: value})
        return getattr(parsed, self._RETURN_MODEL_ATTR)