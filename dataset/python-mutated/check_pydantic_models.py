"""
A script which enforces that Synapse always uses strict types when defining a Pydantic
model.

Pydantic does not yet offer a strict mode, but it is planned for pydantic v2. See

    https://github.com/pydantic/pydantic/issues/1098
    https://pydantic-docs.helpmanual.io/blog/pydantic-v2/#strict-mode

until then, this script is a best effort to stop us from introducing type coersion bugs
(like the infamous stringy power levels fixed in room version 10).
"""
import argparse
import contextlib
import functools
import importlib
import logging
import os
import pkgutil
import sys
import textwrap
import traceback
import unittest.mock
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Set, Type, TypeVar
from parameterized import parameterized
from synapse._pydantic_compat import HAS_PYDANTIC_V2
if TYPE_CHECKING or HAS_PYDANTIC_V2:
    from pydantic.v1 import BaseModel as PydanticBaseModel, conbytes, confloat, conint, constr
    from pydantic.v1.typing import get_args
else:
    from pydantic import BaseModel as PydanticBaseModel, conbytes, confloat, conint, constr
    from pydantic.typing import get_args
from typing_extensions import ParamSpec
logger = logging.getLogger(__name__)
CONSTRAINED_TYPE_FACTORIES_WITH_STRICT_FLAG: List[Callable] = [constr, conbytes, conint, confloat]
TYPES_THAT_PYDANTIC_WILL_COERCE_TO = [str, bytes, int, float, bool]
P = ParamSpec('P')
R = TypeVar('R')

class ModelCheckerException(Exception):
    """Dummy exception. Allows us to detect unwanted types during a module import."""

class MissingStrictInConstrainedTypeException(ModelCheckerException):
    factory_name: str

    def __init__(self, factory_name: str):
        if False:
            return 10
        self.factory_name = factory_name

class FieldHasUnwantedTypeException(ModelCheckerException):
    message: str

    def __init__(self, message: str):
        if False:
            i = 10
            return i + 15
        self.message = message

def make_wrapper(factory: Callable[P, R]) -> Callable[P, R]:
    if False:
        print('Hello World!')
    'We patch `constr` and friends with wrappers that enforce strict=True.'

    @functools.wraps(factory)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if False:
            i = 10
            return i + 15
        if 'strict' not in kwargs:
            raise MissingStrictInConstrainedTypeException(factory.__name__)
        if not kwargs['strict']:
            raise MissingStrictInConstrainedTypeException(factory.__name__)
        return factory(*args, **kwargs)
    return wrapper

def field_type_unwanted(type_: Any) -> bool:
    if False:
        while True:
            i = 10
    'Very rough attempt to detect if a type is unwanted as a Pydantic annotation.\n\n    At present, we exclude types which will coerce, or any generic type involving types\n    which will coerce.'
    logger.debug('Is %s unwanted?')
    if type_ in TYPES_THAT_PYDANTIC_WILL_COERCE_TO:
        logger.debug('yes')
        return True
    logger.debug('Maybe. Subargs are %s', get_args(type_))
    rv = any((field_type_unwanted(t) for t in get_args(type_)))
    logger.debug('Conclusion: %s %s unwanted', type_, 'is' if rv else 'is not')
    return rv

class PatchedBaseModel(PydanticBaseModel):
    """A patched version of BaseModel that inspects fields after models are defined.

    We complain loudly if we see an unwanted type.

    Beware: ModelField.type_ is presumably private; this is likely to be very brittle.
    """

    @classmethod
    def __init_subclass__(cls: Type[PydanticBaseModel], **kwargs: object):
        if False:
            for i in range(10):
                print('nop')
        for field in cls.__fields__.values():
            if field_type_unwanted(field.outer_type_):
                raise FieldHasUnwantedTypeException(f"{cls.__module__}.{cls.__qualname__} has field '{field.name}' with unwanted type `{field.outer_type_}`")

@contextmanager
def monkeypatch_pydantic() -> Generator[None, None, None]:
    if False:
        return 10
    "Patch pydantic with our snooping versions of BaseModel and the con* functions.\n\n    If the snooping functions see something they don't like, they'll raise a\n    ModelCheckingException instance.\n    "
    with contextlib.ExitStack() as patches:
        patch_basemodel1 = unittest.mock.patch('pydantic.BaseModel', new=PatchedBaseModel)
        patch_basemodel2 = unittest.mock.patch('pydantic.main.BaseModel', new=PatchedBaseModel)
        patches.enter_context(patch_basemodel1)
        patches.enter_context(patch_basemodel2)
        for factory in CONSTRAINED_TYPE_FACTORIES_WITH_STRICT_FLAG:
            wrapper: Callable = make_wrapper(factory)
            patch1 = unittest.mock.patch(f'pydantic.{factory.__name__}', new=wrapper)
            patch2 = unittest.mock.patch(f'pydantic.types.{factory.__name__}', new=wrapper)
            patches.enter_context(patch1)
            patches.enter_context(patch2)
        yield

def format_model_checker_exception(e: ModelCheckerException) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Work out which line of code caused e. Format the line in a human-friendly way.'
    if isinstance(e, FieldHasUnwantedTypeException):
        return e.message
    elif isinstance(e, MissingStrictInConstrainedTypeException):
        frame_summary = traceback.extract_tb(e.__traceback__)[-2]
        return f'Missing `strict=True` from {e.factory_name}() call \n' + traceback.format_list([frame_summary])[0].lstrip()
    else:
        raise ValueError(f'Unknown exception {e}') from e

def lint() -> int:
    if False:
        print('Hello World!')
    'Try to import all of Synapse and see if we spot any Pydantic type coercions.\n\n    Print any problems, then return a status code suitable for sys.exit.'
    failures = do_lint()
    if failures:
        print(f'Found {len(failures)} problem(s)')
    for failure in sorted(failures):
        print(failure)
    return os.EX_DATAERR if failures else os.EX_OK

def do_lint() -> Set[str]:
    if False:
        print('Hello World!')
    'Try to import all of Synapse and see if we spot any Pydantic type coercions.'
    failures = set()
    with monkeypatch_pydantic():
        logger.debug('Importing synapse')
        try:
            module = importlib.import_module('synapse')
        except ModelCheckerException as e:
            logger.warning('Bad annotation found when importing synapse')
            failures.add(format_model_checker_exception(e))
            return failures
        try:
            logger.debug('Fetching subpackages')
            module_infos = list(pkgutil.walk_packages(module.__path__, f'{module.__name__}.'))
        except ModelCheckerException as e:
            logger.warning('Bad annotation found when looking for modules to import')
            failures.add(format_model_checker_exception(e))
            return failures
        for module_info in module_infos:
            logger.debug('Importing %s', module_info.name)
            try:
                importlib.import_module(module_info.name)
            except ModelCheckerException as e:
                logger.warning(f'Bad annotation found when importing {module_info.name}')
                failures.add(format_model_checker_exception(e))
    return failures

def run_test_snippet(source: str) -> None:
    if False:
        return 10
    'Exec a snippet of source code in an isolated environment.'
    globals_: Dict[str, object]
    locals_: Dict[str, object]
    globals_ = locals_ = {}
    exec(textwrap.dedent(source), globals_, locals_)

class TestConstrainedTypesPatch(unittest.TestCase):

    def test_expression_without_strict_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import constr\n                except ImportError:\n                    from pydantic import constr\n                constr()\n                ')

    def test_called_as_module_attribute_raises(self) -> None:
        if False:
            while True:
                i = 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                import pydantic\n                pydantic.constr()\n                ')

    def test_wildcard_import_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import *\n                except ImportError:\n                    from pydantic import *\n                constr()\n                ')

    def test_alternative_import_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1.types import constr\n                except ImportError:\n                    from pydantic.types import constr\n                constr()\n                ')

    def test_alternative_import_attribute_raises(self) -> None:
        if False:
            print('Hello World!')
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import types as pydantic_types\n                except ImportError:\n                    from pydantic import types as pydantic_types\n                pydantic_types.constr()\n                ')

    def test_kwarg_but_no_strict_raises(self) -> None:
        if False:
            while True:
                i = 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import constr\n                except ImportError:\n                    from pydantic import constr\n                constr(min_length=10)\n                ')

    def test_kwarg_strict_False_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import constr\n                except ImportError:\n                    from pydantic import constr\n                constr(strict=False)\n                ')

    def test_kwarg_strict_True_doesnt_raise(self) -> None:
        if False:
            i = 10
            return i + 15
        with monkeypatch_pydantic():
            run_test_snippet('\n                try:\n                    from pydantic.v1 import constr\n                except ImportError:\n                    from pydantic import constr\n                constr(strict=True)\n                ')

    def test_annotation_without_strict_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import constr\n                except ImportError:\n                    from pydantic import constr\n                x: constr()\n                ')

    def test_field_annotation_without_strict_raises(self) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1 import BaseModel, conint\n                except ImportError:\n                    from pydantic import BaseModel, conint\n                class C:\n                    x: conint()\n                ')

class TestFieldTypeInspection(unittest.TestCase):

    @parameterized.expand([('str',), 'bytes', ('int',), ('float',), 'bool', ('Optional[str]',), ('Union[None, str]',), ('List[str]',), ('List[List[str]]',), ('Dict[StrictStr, str]',), ('Dict[str, StrictStr]',), ("TypedDict('D', x=int)",)])
    def test_field_holding_unwanted_type_raises(self, annotation: str) -> None:
        if False:
            return 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet(f'\n                from typing import *\n                try:\n                    from pydantic.v1 import *\n                except ImportError:\n                    from pydantic import *\n                class C(BaseModel):\n                    f: {annotation}\n                ')

    @parameterized.expand([('StrictStr',), 'StrictBytes', ('StrictInt',), ('StrictFloat',), 'StrictBool', ('constr(strict=True, min_length=10)',), ('Optional[StrictStr]',), ('Union[None, StrictStr]',), ('List[StrictStr]',), ('List[List[StrictStr]]',), ('Dict[StrictStr, StrictStr]',), ("TypedDict('D', x=StrictInt)",)])
    def test_field_holding_accepted_type_doesnt_raise(self, annotation: str) -> None:
        if False:
            return 10
        with monkeypatch_pydantic():
            run_test_snippet(f'\n                from typing import *\n                try:\n                    from pydantic.v1 import *\n                except ImportError:\n                    from pydantic import *\n                class C(BaseModel):\n                    f: {annotation}\n                ')

    def test_field_holding_str_raises_with_alternative_import(self) -> None:
        if False:
            while True:
                i = 10
        with monkeypatch_pydantic(), self.assertRaises(ModelCheckerException):
            run_test_snippet('\n                try:\n                    from pydantic.v1.main import BaseModel\n                except ImportError:\n                    from pydantic.main import BaseModel\n                class C(BaseModel):\n                    f: str\n                ')
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['lint', 'test'], default='lint', nargs='?')
parser.add_argument('-v', '--verbose', action='store_true')
if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s', level=logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger('xmlschema').setLevel(logging.WARNING)
    if args.mode == 'lint':
        sys.exit(lint())
    elif args.mode == 'test':
        unittest.main(argv=sys.argv[:1])