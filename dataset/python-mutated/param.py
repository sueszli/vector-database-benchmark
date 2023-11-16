from __future__ import annotations
import contextlib
import copy
import datetime
import json
import logging
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, ItemsView, Iterable, MutableMapping, ValuesView
from pendulum.parsing import parse_iso8601
from airflow.exceptions import AirflowException, ParamValidationError, RemovedInAirflow3Warning
from airflow.utils import timezone
from airflow.utils.mixins import ResolveMixin
from airflow.utils.types import NOTSET, ArgNotSet
if TYPE_CHECKING:
    from airflow.models.dag import DAG
    from airflow.models.dagrun import DagRun
    from airflow.models.operator import Operator
    from airflow.serialization.pydantic.dag_run import DagRunPydantic
    from airflow.utils.context import Context
logger = logging.getLogger(__name__)

class Param:
    """
    Class to hold the default value of a Param and rule set to do the validations.

    Without the rule set it always validates and returns the default value.

    :param default: The value this Param object holds
    :param description: Optional help text for the Param
    :param schema: The validation schema of the Param, if not given then all kwargs except
        default & description will form the schema
    """
    __version__: ClassVar[int] = 1
    CLASS_IDENTIFIER = '__class'

    def __init__(self, default: Any=NOTSET, description: str | None=None, **kwargs):
        if False:
            return 10
        if default is not NOTSET:
            self._warn_if_not_json(default)
        self.value = default
        self.description = description
        self.schema = kwargs.pop('schema') if 'schema' in kwargs else kwargs

    def __copy__(self) -> Param:
        if False:
            for i in range(10):
                print('nop')
        return Param(self.value, self.description, schema=self.schema)

    @staticmethod
    def _warn_if_not_json(value):
        if False:
            return 10
        try:
            json.dumps(value)
        except Exception:
            warnings.warn('The use of non-json-serializable params is deprecated and will be removed in a future release', RemovedInAirflow3Warning)

    @staticmethod
    def _warn_if_not_rfc3339_dt(value):
        if False:
            while True:
                i = 10
        'Fallback to iso8601 datetime validation if rfc3339 failed.'
        try:
            iso8601_value = parse_iso8601(value)
        except Exception:
            return None
        if not isinstance(iso8601_value, datetime.datetime):
            return None
        warnings.warn(f'The use of non-RFC3339 datetime: {value!r} is deprecated and will be removed in a future release', RemovedInAirflow3Warning)
        if timezone.is_naive(iso8601_value):
            warnings.warn('The use naive datetime is deprecated and will be removed in a future release', RemovedInAirflow3Warning)
        return value

    def resolve(self, value: Any=NOTSET, suppress_exception: bool=False) -> Any:
        if False:
            while True:
                i = 10
        "\n        Run the validations and returns the Param's final value.\n\n        May raise ValueError on failed validations, or TypeError\n        if no value is passed and no value already exists.\n        We first check that value is json-serializable; if not, warn.\n        In future release we will require the value to be json-serializable.\n\n        :param value: The value to be updated for the Param\n        :param suppress_exception: To raise an exception or not when the validations fails.\n            If true and validations fails, the return value would be None.\n        "
        import jsonschema
        from jsonschema import FormatChecker
        from jsonschema.exceptions import ValidationError
        if value is not NOTSET:
            self._warn_if_not_json(value)
        final_val = self.value if value is NOTSET else value
        if isinstance(final_val, ArgNotSet):
            if suppress_exception:
                return None
            raise ParamValidationError('No value passed and Param has no default value')
        try:
            jsonschema.validate(final_val, self.schema, format_checker=FormatChecker())
        except ValidationError as err:
            if err.schema.get('format') == 'date-time':
                rfc3339_value = self._warn_if_not_rfc3339_dt(final_val)
                if rfc3339_value:
                    self.value = rfc3339_value
                    return rfc3339_value
            if suppress_exception:
                return None
            raise ParamValidationError(err) from None
        self.value = final_val
        return final_val

    def dump(self) -> dict:
        if False:
            i = 10
            return i + 15
        'Dump the Param as a dictionary.'
        out_dict: dict[str, str | None] = {self.CLASS_IDENTIFIER: f'{self.__module__}.{self.__class__.__name__}'}
        out_dict.update(self.__dict__)
        if self.value is NOTSET:
            out_dict['value'] = None
        return out_dict

    @property
    def has_value(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.value is not NOTSET and self.value is not None

    def serialize(self) -> dict:
        if False:
            return 10
        return {'value': self.value, 'description': self.description, 'schema': self.schema}

    @staticmethod
    def deserialize(data: dict[str, Any], version: int) -> Param:
        if False:
            i = 10
            return i + 15
        if version > Param.__version__:
            raise TypeError('serialized version > class version')
        return Param(default=data['value'], description=data['description'], schema=data['schema'])

class ParamsDict(MutableMapping[str, Any]):
    """
    Class to hold all params for dags or tasks.

    All the keys are strictly string and values are converted into Param's object
    if they are not already. This class is to replace param's dictionary implicitly
    and ideally not needed to be used directly.


    :param dict_obj: A dict or dict like object to init ParamsDict
    :param suppress_exception: Flag to suppress value exceptions while initializing the ParamsDict
    """
    __version__: ClassVar[int] = 1
    __slots__ = ['__dict', 'suppress_exception']

    def __init__(self, dict_obj: MutableMapping | None=None, suppress_exception: bool=False):
        if False:
            print('Hello World!')
        params_dict: dict[str, Param] = {}
        dict_obj = dict_obj or {}
        for (k, v) in dict_obj.items():
            if not isinstance(v, Param):
                params_dict[k] = Param(v)
            else:
                params_dict[k] = v
        self.__dict = params_dict
        self.suppress_exception = suppress_exception

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self.__dict)

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ParamsDict):
            return self.dump() == other.dump()
        if isinstance(other, dict):
            return self.dump() == other
        return NotImplemented

    def __copy__(self) -> ParamsDict:
        if False:
            print('Hello World!')
        return ParamsDict(self.__dict, self.suppress_exception)

    def __deepcopy__(self, memo: dict[int, Any] | None) -> ParamsDict:
        if False:
            i = 10
            return i + 15
        return ParamsDict(copy.deepcopy(self.__dict, memo), self.suppress_exception)

    def __contains__(self, o: object) -> bool:
        if False:
            while True:
                i = 10
        return o in self.__dict

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.__dict)

    def __delitem__(self, v: str) -> None:
        if False:
            print('Hello World!')
        del self.__dict[v]

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.__dict)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return repr(self.dump())

    def __setitem__(self, key: str, value: Any) -> None:
        if False:
            while True:
                i = 10
        "\n        Override for dictionary's ``setitem`` method to ensure all values are of Param's type only.\n\n        :param key: A key which needs to be inserted or updated in the dict\n        :param value: A value which needs to be set against the key. It could be of any\n            type but will be converted and stored as a Param object eventually.\n        "
        if isinstance(value, Param):
            param = value
        elif key in self.__dict:
            param = self.__dict[key]
            try:
                param.resolve(value=value, suppress_exception=self.suppress_exception)
            except ParamValidationError as ve:
                raise ParamValidationError(f'Invalid input for param {key}: {ve}') from None
        else:
            param = Param(value)
        self.__dict[key] = param

    def __getitem__(self, key: str) -> Any:
        if False:
            i = 10
            return i + 15
        "\n        Override for dictionary's ``getitem`` method to call the resolve method after fetching the key.\n\n        :param key: The key to fetch\n        "
        param = self.__dict[key]
        return param.resolve(suppress_exception=self.suppress_exception)

    def get_param(self, key: str) -> Param:
        if False:
            i = 10
            return i + 15
        'Get the internal :class:`.Param` object for this key.'
        return self.__dict[key]

    def items(self):
        if False:
            i = 10
            return i + 15
        return ItemsView(self.__dict)

    def values(self):
        if False:
            for i in range(10):
                print('nop')
        return ValuesView(self.__dict)

    def update(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        if len(args) == 1 and (not kwargs) and isinstance(args[0], ParamsDict):
            return super().update(args[0].__dict)
        super().update(*args, **kwargs)

    def dump(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Dump the ParamsDict object as a dictionary, while suppressing exceptions.'
        return {k: v.resolve(suppress_exception=True) for (k, v) in self.items()}

    def validate(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Validate & returns all the Params object stored in the dictionary.'
        resolved_dict = {}
        try:
            for (k, v) in self.items():
                resolved_dict[k] = v.resolve(suppress_exception=self.suppress_exception)
        except ParamValidationError as ve:
            raise ParamValidationError(f'Invalid input for param {k}: {ve}') from None
        return resolved_dict

    def serialize(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self.dump()

    @staticmethod
    def deserialize(data: dict, version: int) -> ParamsDict:
        if False:
            for i in range(10):
                print('nop')
        if version > ParamsDict.__version__:
            raise TypeError('serialized version > class version')
        return ParamsDict(data)

class DagParam(ResolveMixin):
    """DAG run parameter reference.

    This binds a simple Param object to a name within a DAG instance, so that it
    can be resolved during the runtime via the ``{{ context }}`` dictionary. The
    ideal use case of this class is to implicitly convert args passed to a
    method decorated by ``@dag``.

    It can be used to parameterize a DAG. You can overwrite its value by setting
    it on conf when you trigger your DagRun.

    This can also be used in templates by accessing ``{{ context.params }}``.

    **Example**:

        with DAG(...) as dag:
          EmailOperator(subject=dag.param('subject', 'Hi from Airflow!'))

    :param current_dag: Dag being used for parameter.
    :param name: key value which is used to set the parameter
    :param default: Default value used if no parameter was set.
    """

    def __init__(self, current_dag: DAG, name: str, default: Any=NOTSET):
        if False:
            i = 10
            return i + 15
        if default is not NOTSET:
            current_dag.params[name] = default
        self._name = name
        self._default = default

    def iter_references(self) -> Iterable[tuple[Operator, str]]:
        if False:
            print('Hello World!')
        return ()

    def resolve(self, context: Context) -> Any:
        if False:
            return 10
        'Pull DagParam value from DagRun context. This method is run during ``op.execute()``.'
        with contextlib.suppress(KeyError):
            return context['dag_run'].conf[self._name]
        if self._default is not NOTSET:
            return self._default
        with contextlib.suppress(KeyError):
            return context['params'][self._name]
        raise AirflowException(f'No value could be resolved for parameter {self._name}')

def process_params(dag: DAG, task: Operator, dag_run: DagRun | DagRunPydantic | None, *, suppress_exception: bool) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Merge, validate params, and convert them into a simple dict.'
    from airflow.configuration import conf
    params = ParamsDict(suppress_exception=suppress_exception)
    with contextlib.suppress(AttributeError):
        params.update(dag.params)
    if task.params:
        params.update(task.params)
    if conf.getboolean('core', 'dag_run_conf_overrides_params') and dag_run and dag_run.conf:
        logger.debug('Updating task params (%s) with DagRun.conf (%s)', params, dag_run.conf)
        params.update(dag_run.conf)
    return params.validate()