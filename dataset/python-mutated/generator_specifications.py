from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Optional, Set
import re
from .parameter import Parameter

class ParameterAnnotation(ABC):
    """Parameter annotations can either be a uniform string or a mapping from parameter
    name to annotation."""

    @abstractmethod
    def get(self, parameter: 'Parameter') -> Optional[str]:
        if False:
            while True:
                i = 10
        pass

class PerParameterAnnotation(ParameterAnnotation):

    def __init__(self, parameter_name_to_taint: Dict[str, str]) -> None:
        if False:
            while True:
                i = 10
        self.parameter_name_to_taint = parameter_name_to_taint

    def get(self, parameter: 'Parameter') -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.parameter_name_to_taint.get(parameter.name)

class AllParametersAnnotation(ParameterAnnotation):

    def __init__(self, arg: Optional[str]=None, vararg: Optional[str]=None, kwarg: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        self.arg = arg
        self.kwarg = kwarg
        self.vararg = vararg

    def get(self, parameter: 'Parameter') -> Optional[str]:
        if False:
            print('Hello World!')
        if parameter.kind == Parameter.Kind.ARG:
            return self.arg
        elif parameter.kind == Parameter.Kind.VARARG:
            return self.vararg
        else:
            return self.kwarg

class AllParametersAnnotationWithParameterNameAsSubKind(ParameterAnnotation):

    def __init__(self, parameter_taint: str, parameter_kind: str) -> None:
        if False:
            return 10
        self.parameter_taint = parameter_taint
        self.parameter_kind = parameter_kind

    def get(self, parameter: 'Parameter') -> Optional[str]:
        if False:
            print('Hello World!')
        sanitized_parameter_name = re.compile('[^a-zA-Z_0-9]').sub('', parameter.name)
        return f'{self.parameter_kind}[{self.parameter_taint}[{sanitized_parameter_name}]]'

class AnnotationSpecification(NamedTuple):
    parameter_annotation: Optional[ParameterAnnotation] = None
    returns: Optional[str] = None

class WhitelistSpecification(NamedTuple):

    def __hash__(self) -> int:
        if False:
            return 10
        parameter_type = self.parameter_type
        parameter_name = self.parameter_name
        return hash((parameter_type and tuple(sorted(parameter_type)), parameter_name and tuple(sorted(parameter_name))))
    parameter_type: Optional[Set[str]] = None
    parameter_name: Optional[Set[str]] = None

class DecoratorAnnotationSpecification(NamedTuple):

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash((self.decorator, self.annotations, self.whitelist))
    decorator: str
    annotations: Optional[AnnotationSpecification] = None
    whitelist: Optional[WhitelistSpecification] = None
default_entrypoint_taint = AnnotationSpecification(parameter_annotation=AllParametersAnnotation(arg='TaintSource[UserControlled]', vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]'), returns='TaintSink[ReturnedToUser]')