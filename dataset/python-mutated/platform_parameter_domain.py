"""Domain objects for platform parameters."""
from __future__ import annotations
import enum
import json
import re
from core import feconf
from core import platform_feature_list
from core import utils
from core.constants import constants
from core.domain import change_domain
from typing import Callable, Dict, Final, List, Optional, Pattern, TypedDict, Union

class ServerMode(enum.Enum):
    """Enum for server modes."""
    DEV = 'dev'
    TEST = 'test'
    PROD = 'prod'
FeatureStages = ServerMode
PlatformDataTypes = Union[str, int, bool, float]

class DataTypes(enum.Enum):
    """Enum for data types."""
    BOOL = 'bool'
    STRING = 'string'
    NUMBER = 'number'
ALLOWED_SERVER_MODES: Final = [ServerMode.DEV.value, ServerMode.TEST.value, ServerMode.PROD.value]
ALLOWED_FEATURE_STAGES: Final = [FeatureStages.DEV.value, FeatureStages.TEST.value, FeatureStages.PROD.value]
ALLOWED_PLATFORM_TYPES: List[str] = constants.PLATFORM_PARAMETER_ALLOWED_PLATFORM_TYPES
ALLOWED_APP_VERSION_FLAVORS: List[str] = constants.PLATFORM_PARAMETER_ALLOWED_APP_VERSION_FLAVORS
APP_VERSION_WITH_HASH_REGEXP: Pattern[str] = re.compile(constants.PLATFORM_PARAMETER_APP_VERSION_WITH_HASH_REGEXP)
APP_VERSION_WITHOUT_HASH_REGEXP: Pattern[str] = re.compile(constants.PLATFORM_PARAMETER_APP_VERSION_WITHOUT_HASH_REGEXP)

class PlatformParameterChange(change_domain.BaseChange):
    """Domain object for changes made to a platform parameter object.

    The allowed commands, together with the attributes:
        - 'edit_rules' (with new_rules)
    """
    CMD_EDIT_RULES: Final = 'edit_rules'
    ALLOWED_COMMANDS: List[feconf.ValidCmdDict] = [{'name': CMD_EDIT_RULES, 'required_attribute_names': ['new_rules'], 'optional_attribute_names': [], 'user_id_attribute_names': [], 'allowed_values': {}, 'deprecated_values': {}}]

class EditRulesPlatformParameterCmd(PlatformParameterChange):
    """Class representing the PlatformParameterChange's
    CMD_EDIT_RULES command.
    """
    new_rules: List[str]

class ClientSideContextDict(TypedDict):
    """Dictionary representing the client's side Context object."""
    platform_type: Optional[str]
    app_version: Optional[str]

class ServerSideContextDict(TypedDict):
    """Dictionary representing the server's side Context object."""
    server_mode: ServerMode

class EvaluationContext:
    """Domain object representing the context for parameter evaluation."""

    def __init__(self, platform_type: Optional[str], app_version: Optional[str], server_mode: ServerMode) -> None:
        if False:
            print('Hello World!')
        self._platform_type = platform_type
        self._app_version = app_version
        self._server_mode = server_mode

    @property
    def platform_type(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        "Returns platform type.\n\n        Returns:\n            str|None. The platform type, e.g. 'Web', 'Android', 'Backend'.\n        "
        return self._platform_type

    @property
    def app_version(self) -> Optional[str]:
        if False:
            print('Hello World!')
        "Returns client application version.\n\n        Returns:\n            str|None. The version of native application, e.g. '1.0.0',\n            None if the platform type is Web.\n        "
        return self._app_version

    @property
    def server_mode(self) -> ServerMode:
        if False:
            i = 10
            return i + 15
        'Returns the server mode of Oppia.\n\n        Returns:\n            Enum(ServerMode). The the server mode of Oppia,\n            must be one of the following: dev, test, prod.\n        '
        return self._server_mode

    @property
    def is_valid(self) -> bool:
        if False:
            return 10
        "Returns whether this context object is valid for evaluating\n        parameters. An invalid context object usually indicates that one of the\n        object's required fields is missing or an unexpected value. Note that\n        objects which are not valid will still pass validation. This method\n        should return true and validate() should not raise an exception before\n        using this object for platform evaluation.\n\n        Returns:\n            bool. Whether this context object can be used for evaluating\n            parameters.\n        "
        return self._platform_type is not None and self._platform_type in ALLOWED_PLATFORM_TYPES

    def validate(self) -> None:
        if False:
            return 10
        'Validates the EvaluationContext domain object, raising an exception\n        if the object is in an irrecoverable error state.\n        '
        if self._app_version is not None:
            match = APP_VERSION_WITH_HASH_REGEXP.match(self._app_version)
            if match is None:
                raise utils.ValidationError("Invalid version '%s', expected to match regexp %s." % (self._app_version, APP_VERSION_WITH_HASH_REGEXP))
            if match.group(2) is not None and match.group(2) not in ALLOWED_APP_VERSION_FLAVORS:
                raise utils.ValidationError("Invalid version flavor '%s', must be one of %s if specified." % (match.group(2), ALLOWED_APP_VERSION_FLAVORS))
        if self._server_mode.value not in ALLOWED_SERVER_MODES:
            raise utils.ValidationError("Invalid server mode '%s', must be one of %s." % (self._server_mode.value, ALLOWED_SERVER_MODES))

    @classmethod
    def from_dict(cls, client_context_dict: ClientSideContextDict, server_context_dict: ServerSideContextDict) -> EvaluationContext:
        if False:
            while True:
                i = 10
        'Creates a new EvaluationContext object by combining both client side\n        and server side context.\n\n        Args:\n            client_context_dict: dict. The client side context.\n            server_context_dict: dict. The server side context.\n\n        Returns:\n            EvaluationContext. The corresponding EvaluationContext domain\n            object.\n        '
        return cls(client_context_dict['platform_type'], client_context_dict.get('app_version'), server_context_dict['server_mode'])

class PlatformParameterFilterDict(TypedDict):
    """Dictionary representing the PlatformParameterFilter object."""
    type: str
    conditions: List[List[str]]

class PlatformParameterFilter:
    """Domain object for filters in platform parameters."""
    SUPPORTED_FILTER_TYPES: Final = ['server_mode', 'platform_type', 'app_version', 'app_version_flavor']
    SUPPORTED_OP_FOR_FILTERS: Final = {'platform_type': ['='], 'app_version_flavor': ['=', '<', '<=', '>', '>='], 'app_version': ['=', '<', '<=', '>', '>=']}

    def __init__(self, filter_type: str, conditions: List[List[str]]) -> None:
        if False:
            return 10
        self._type = filter_type
        self._conditions = conditions

    @property
    def type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns filter type.\n\n        Returns:\n            str. The filter type.\n        '
        return self._type

    @property
    def conditions(self) -> List[List[str]]:
        if False:
            i = 10
            return i + 15
        'Returns filter conditions.\n\n        Returns:\n            list(list(str)). The filter conditions. Each element of the list\n            contain a list with 2-elements [op, value], where op is the operator\n            for comparison, value is the value used for comparison.\n        '
        return self._conditions

    def evaluate(self, context: EvaluationContext) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Tries to match the given context with the filter against its\n        value(s).\n\n        Args:\n            context: EvaluationContext. The context for evaluation.\n\n        Returns:\n            bool. True if the filter is matched.\n        '
        return any((self._evaluate_single_value(op, value, context) for (op, value) in self._conditions))

    def _evaluate_single_value(self, op: str, value: str, context: EvaluationContext) -> bool:
        if False:
            i = 10
            return i + 15
        "Tries to match the given context with the filter against the\n        given value.\n\n        Args:\n            op: str. The operator for comparison, e.g. '='.\n            value: str. The value to match against.\n            context: EvaluationContext. The context for evaluation.\n\n        Returns:\n            bool. True if the filter is matched.\n\n        Raises:\n            Exception. Given operator is not supported.\n        "
        if self._type == 'platform_type' and op != '=':
            raise Exception("Unsupported comparison operator '%s' for %s filter, expected one of %s." % (op, self._type, self.SUPPORTED_OP_FOR_FILTERS[self._type]))
        matched = False
        if self._type == 'platform_type' and op == '=':
            matched = context.platform_type == value
        elif self._type == 'app_version_flavor':
            assert context.app_version is not None
            matched = self._match_version_flavor(op, value, context.app_version)
        elif self._type == 'app_version':
            matched = self._match_version_expression(op, value, context.app_version)
        return matched

    def validate(self) -> None:
        if False:
            i = 10
            return i + 15
        'Validates the PlatformParameterFilter domain object.'
        if self._type not in self.SUPPORTED_FILTER_TYPES:
            raise utils.ValidationError("Unsupported filter type '%s'" % self._type)
        for (op, _) in self._conditions:
            if op not in self.SUPPORTED_OP_FOR_FILTERS[self._type]:
                raise utils.ValidationError("Unsupported comparison operator '%s' for %s filter, expected one of %s." % (op, self._type, self.SUPPORTED_OP_FOR_FILTERS[self._type]))
        if self._type == 'platform_type':
            for (_, platform_type) in self._conditions:
                if platform_type not in ALLOWED_PLATFORM_TYPES:
                    raise utils.ValidationError("Invalid platform type '%s', must be one of %s." % (platform_type, ALLOWED_PLATFORM_TYPES))
        elif self._type == 'app_version_flavor':
            for (_, flavor) in self._conditions:
                if flavor not in ALLOWED_APP_VERSION_FLAVORS:
                    raise utils.ValidationError("Invalid app version flavor '%s', must be one of %s." % (flavor, ALLOWED_APP_VERSION_FLAVORS))
        elif self._type == 'app_version':
            for (_, version) in self._conditions:
                if not APP_VERSION_WITHOUT_HASH_REGEXP.match(version):
                    raise utils.ValidationError("Invalid version expression '%s', expected to matchregexp %s." % (version, APP_VERSION_WITHOUT_HASH_REGEXP))

    def to_dict(self) -> PlatformParameterFilterDict:
        if False:
            return 10
        'Returns a dict representation of the PlatformParameterFilter domain\n        object.\n\n        Returns:\n            dict. A dict mapping of all fields of PlatformParameterFilter\n            object.\n        '
        return {'type': self._type, 'conditions': self._conditions}

    @classmethod
    def from_dict(cls, filter_dict: PlatformParameterFilterDict) -> PlatformParameterFilter:
        if False:
            print('Hello World!')
        'Returns an PlatformParameterFilter object from a dict.\n\n        Args:\n            filter_dict: dict. A dict mapping of all fields of\n                PlatformParameterFilter object.\n\n        Returns:\n            PlatformParameterFilter. The corresponding PlatformParameterFilter\n            domain object.\n        '
        return cls(filter_dict['type'], filter_dict['conditions'])

    def _match_version_expression(self, op: str, value: str, client_version: Optional[str]) -> bool:
        if False:
            while True:
                i = 10
        "Tries to match the version expression against the client version.\n\n        Args:\n            op: str. The operator for comparison, e.g. '=', '>'.\n            value: str. The version for comparison, e.g. '1.0.1'.\n            client_version: str|None. The client version, e.g. '1.0.1-3aebf3h'.\n\n        Returns:\n            bool. True if the expression matches the version.\n\n        Raises:\n            Exception. Given operator is not supported.\n        "
        if client_version is None:
            return False
        match = APP_VERSION_WITH_HASH_REGEXP.match(client_version)
        assert match is not None
        client_version_without_hash = match.group(1)
        is_equal = value == client_version_without_hash
        is_client_version_smaller = self._is_first_version_smaller(client_version_without_hash, value)
        is_client_version_larger = self._is_first_version_smaller(value, client_version_without_hash)
        if op == '=':
            return is_equal
        elif op == '<':
            return is_client_version_smaller
        elif op == '<=':
            return is_equal or is_client_version_smaller
        elif op == '>':
            return is_client_version_larger
        elif op == '>=':
            return is_equal or is_client_version_larger
        else:
            raise Exception("Unsupported comparison operator '%s' for %s filter, expected one of %s." % (op, self._type, self.SUPPORTED_OP_FOR_FILTERS[self._type]))

    def _is_first_version_smaller(self, version_a: str, version_b: str) -> bool:
        if False:
            return 10
        "Compares two version strings, return True if the first version is\n        smaller.\n\n        Args:\n            version_a: str. The version string (e.g. '1.0.0').\n            version_b: str. The version string (e.g. '1.0.0').\n\n        Returns:\n            bool. True if the first version is smaller.\n        "
        splitted_version_a = version_a.split('.')
        splitted_version_b = version_b.split('.')
        for (sub_version_a, sub_version_b) in zip(splitted_version_a, splitted_version_b):
            if int(sub_version_a) < int(sub_version_b):
                return True
            elif int(sub_version_a) > int(sub_version_b):
                return False
        return False

    def _match_version_flavor(self, op: str, flavor: str, client_version: str) -> bool:
        if False:
            print('Hello World!')
        "Matches the client version flavor.\n\n        Args:\n            op: str. The operator for comparison, e.g. '=', '>'.\n            flavor: str. The flavor to match, e.g. 'alpha', 'beta', 'test',\n                'release'.\n            client_version: str. The version of the client, given in the form\n                of '<version>-<hash>-<flavor>'. The hash and flavor of client\n                version is optional, but if absent, no flavor filter will\n                match to it.\n\n        Returns:\n            bool. True is the client_version matches the given flavor using\n            the operator.\n\n        Raises:\n            Exception. Given operator is not supported.\n        "
        match = APP_VERSION_WITH_HASH_REGEXP.match(client_version)
        assert match is not None
        client_flavor = match.group(2)
        if client_flavor is None:
            return False
        is_equal = flavor == client_flavor
        is_client_flavor_smaller = self._is_first_flavor_smaller(client_flavor, flavor)
        is_client_flavor_larger = self._is_first_flavor_smaller(flavor, client_flavor)
        if op == '=':
            return is_equal
        elif op == '<':
            return is_client_flavor_smaller
        elif op == '<=':
            return is_equal or is_client_flavor_smaller
        elif op == '>':
            return is_client_flavor_larger
        elif op == '>=':
            return is_equal or is_client_flavor_larger
        else:
            raise Exception("Unsupported comparison operator '%s' for %s filter, expected one of %s." % (op, self._type, self.SUPPORTED_OP_FOR_FILTERS[self._type]))

    def _is_first_flavor_smaller(self, flavor_a: str, flavor_b: str) -> bool:
        if False:
            while True:
                i = 10
        "Compares two version flavors, return True if the first version is\n        smaller in the following ordering:\n        'test' < 'alpha' < 'beta' < 'release'.\n\n        Args:\n            flavor_a: str. The version flavor.\n            flavor_b: str. The version flavor.\n\n        Returns:\n            bool. True if the first flavor is smaller.\n        "
        return ALLOWED_APP_VERSION_FLAVORS.index(flavor_a) < ALLOWED_APP_VERSION_FLAVORS.index(flavor_b)

class PlatformParameterRuleDict(TypedDict):
    """Dictionary representing the PlatformParameterRule object."""
    filters: List[PlatformParameterFilterDict]
    value_when_matched: PlatformDataTypes

class PlatformParameterRule:
    """Domain object for rules in platform parameters."""

    def __init__(self, filters: List[PlatformParameterFilter], value_when_matched: PlatformDataTypes) -> None:
        if False:
            while True:
                i = 10
        self._filters = filters
        self._value_when_matched = value_when_matched

    @property
    def filters(self) -> List[PlatformParameterFilter]:
        if False:
            return 10
        'Returns the filters of the rule.\n\n        Returns:\n            list(PlatformParameterFilter). The filters of the rule.\n        '
        return self._filters

    @property
    def value_when_matched(self) -> PlatformDataTypes:
        if False:
            while True:
                i = 10
        'Returns the value outcome if this rule is matched.\n\n        Returns:\n            *. The value outcome.\n        '
        return self._value_when_matched

    def evaluate(self, context: EvaluationContext) -> bool:
        if False:
            while True:
                i = 10
        'Tries to match the given context with the rule against its filter(s).\n        A rule is matched when all its filters are matched.\n\n        Args:\n            context: EvaluationContext. The context for evaluation.\n\n        Returns:\n            bool. True if the rule is matched.\n        '
        return all((filter_domain.evaluate(context) for filter_domain in self._filters))

    def to_dict(self) -> PlatformParameterRuleDict:
        if False:
            return 10
        'Returns a dict representation of the PlatformParameterRule domain\n        object.\n\n        Returns:\n            dict. A dict mapping of all fields of PlatformParameterRule\n            object.\n        '
        return {'filters': [filter_domain.to_dict() for filter_domain in self._filters], 'value_when_matched': self._value_when_matched}

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the PlatformParameterRule domain object.'
        for filter_domain_object in self._filters:
            filter_domain_object.validate()

    @classmethod
    def from_dict(cls, rule_dict: PlatformParameterRuleDict) -> PlatformParameterRule:
        if False:
            i = 10
            return i + 15
        'Returns an PlatformParameterRule object from a dict.\n\n        Args:\n            rule_dict: dict. A dict mapping of all fields of\n                PlatformParameterRule object.\n\n        Returns:\n            PlatformParameterRule. The corresponding PlatformParameterRule\n            domain object.\n        '
        return cls([PlatformParameterFilter.from_dict(filter_dict) for filter_dict in rule_dict['filters']], rule_dict['value_when_matched'])

class PlatformParameterDict(TypedDict):
    """Dictionary representing the PlatformParameter object."""
    name: str
    description: str
    data_type: str
    rules: List[PlatformParameterRuleDict]
    rule_schema_version: int
    default_value: PlatformDataTypes
    is_feature: bool
    feature_stage: Optional[str]

class PlatformParameter:
    """Domain object for platform parameters."""
    DATA_TYPE_PREDICATES_DICT: Dict[str, Callable[[PlatformDataTypes], bool]] = {DataTypes.BOOL.value: lambda x: isinstance(x, bool), DataTypes.STRING.value: lambda x: isinstance(x, str), DataTypes.NUMBER.value: lambda x: isinstance(x, (float, int))}
    PARAMETER_NAME_REGEXP: Final = '^[A-Za-z0-9_]{1,100}$'

    def __init__(self, name: str, description: str, data_type: str, rules: List[PlatformParameterRule], rule_schema_version: int, default_value: PlatformDataTypes, is_feature: bool, feature_stage: Optional[str]) -> None:
        if False:
            return 10
        self._name = name
        self._description = description
        self._data_type = data_type
        self._rules = rules
        self._rule_schema_version = rule_schema_version
        self._default_value = default_value
        self._is_feature = is_feature
        self._feature_stage = feature_stage

    @property
    def name(self) -> str:
        if False:
            return 10
        'Returns the name of the platform parameter.\n\n        Returns:\n            str. The name of the platform parameter.\n        '
        return self._name

    @property
    def description(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the description of the platform parameter.\n\n        Returns:\n            str. The description of the platform parameter.\n        '
        return self._description

    @property
    def data_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the data type of the platform parameter.\n\n        Returns:\n            DATA_TYPES. The data type of the platform parameter.\n        '
        return self._data_type

    @property
    def rules(self) -> List[PlatformParameterRule]:
        if False:
            i = 10
            return i + 15
        'Returns the rules of the platform parameter.\n\n        Returns:\n            list(PlatformParameterRules). The rules of the platform parameter.\n        '
        return self._rules

    def set_rules(self, new_rules: List[PlatformParameterRule]) -> None:
        if False:
            print('Hello World!')
        'Sets the rules of the PlatformParameter.\n\n        Args:\n            new_rules: list(PlatformParameterRules). The new rules of the\n                parameter.\n        '
        self._rules = new_rules

    @property
    def rule_schema_version(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the schema version of the rules.\n\n        Returns:\n            int. The schema version of the rules.\n        '
        return self._rule_schema_version

    @property
    def default_value(self) -> PlatformDataTypes:
        if False:
            i = 10
            return i + 15
        'Returns the default value of the platform parameter.\n\n        Returns:\n            *. The default value of the platform parameter.\n        '
        return self._default_value

    def set_default_value(self, default_value: PlatformDataTypes) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the default value of the PlatformParameter.\n\n        Args:\n            default_value: PlatformDataTypes. The new default value of the\n                parameter.\n        '
        self._default_value = default_value

    @property
    def is_feature(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns whether this parameter is also a feature flag.\n\n        Returns:\n            bool. True if the parameter is a feature flag.\n        '
        return self._is_feature

    @property
    def feature_stage(self) -> Optional[str]:
        if False:
            print('Hello World!')
        "Returns the stage of the feature flag.\n\n        Returns:\n            FeatureStages|None. The stage of the feature flag, None if the\n            parameter isn't a feature flag.\n        "
        return self._feature_stage

    def validate(self) -> None:
        if False:
            print('Hello World!')
        'Validates the PlatformParameter domain object.'
        if re.match(self.PARAMETER_NAME_REGEXP, self._name) is None:
            raise utils.ValidationError("Invalid parameter name '%s', expected to match regexp %s." % (self._name, self.PARAMETER_NAME_REGEXP))
        if self._data_type not in self.DATA_TYPE_PREDICATES_DICT:
            raise utils.ValidationError("Unsupported data type '%s'." % self._data_type)
        all_platform_params_names = [param.value for param in platform_feature_list.ALL_PLATFORM_PARAMS_EXCEPT_FEATURE_FLAGS]
        if self._feature_stage is not None and self._name in all_platform_params_names:
            raise utils.ValidationError('The feature stage of the platform parameter %s should be None.' % self._name)
        predicate = self.DATA_TYPE_PREDICATES_DICT[self.data_type]
        if not predicate(self._default_value):
            raise utils.ValidationError("Expected %s, received '%s' in default value." % (self._data_type, self._default_value))
        for rule in self._rules:
            if not predicate(rule.value_when_matched):
                raise utils.ValidationError("Expected %s, received '%s' in value_when_matched." % (self._data_type, rule.value_when_matched))
            rule.validate()
        if self._is_feature:
            self._validate_feature_flag()

    def _get_server_mode(self) -> ServerMode:
        if False:
            while True:
                i = 10
        'Returns the current server mode.\n\n        Returns:\n            ServerMode. The current server mode.\n        '
        return ServerMode.DEV if constants.DEV_MODE else ServerMode.PROD if feconf.ENV_IS_OPPIA_ORG_PRODUCTION_SERVER else ServerMode.TEST

    def evaluate(self, context: EvaluationContext) -> PlatformDataTypes:
        if False:
            return 10
        'Evaluates the value of the platform parameter in the given context.\n        The value of first matched rule is returned as the result.\n\n        Note that if the provided context is in an invalid state (e.g. its\n        is_valid property returns false) then this parameter will defer to its\n        default value since it may not be safe to partially evaluate the\n        parameter for an unrecognized or partially recognized context.\n\n        Args:\n            context: EvaluationContext. The context for evaluation.\n\n        Returns:\n            *. The evaluate result of the platform parameter.\n        '
        if context.is_valid:
            if self._is_feature:
                server_mode = self._get_server_mode()
                if server_mode == ServerMode.TEST and self._feature_stage == ServerMode.DEV.value:
                    return False
                if server_mode == ServerMode.PROD and self._feature_stage in (ServerMode.DEV.value, ServerMode.TEST.value):
                    return False
            for rule in self._rules:
                if rule.evaluate(context):
                    return rule.value_when_matched
        return self._default_value

    def to_dict(self) -> PlatformParameterDict:
        if False:
            return 10
        'Returns a dict representation of the PlatformParameter domain\n        object.\n\n        Returns:\n            dict. A dict mapping of all fields of PlatformParameter object.\n        '
        return {'name': self._name, 'description': self._description, 'data_type': self._data_type, 'rules': [rule.to_dict() for rule in self._rules], 'rule_schema_version': self._rule_schema_version, 'default_value': self._default_value, 'is_feature': self._is_feature, 'feature_stage': self._feature_stage}

    def _validate_feature_flag(self) -> None:
        if False:
            while True:
                i = 10
        'Validates the PlatformParameter domain object that is a feature\n        flag.\n        '
        if self._default_value is True:
            raise utils.ValidationError('Feature flag is not allowed to have default value as True.')
        if self._data_type != DataTypes.BOOL.value:
            raise utils.ValidationError("Data type of feature flags must be bool, got '%s' instead." % self._data_type)
        if not any((self._feature_stage == feature_stage for feature_stage in ALLOWED_FEATURE_STAGES)):
            raise utils.ValidationError("Invalid feature stage, got '%s', expected one of %s." % (self._feature_stage, ALLOWED_FEATURE_STAGES))
        server_mode = self._get_server_mode()
        if server_mode == ServerMode.TEST and self._feature_stage == ServerMode.DEV.value:
            raise utils.ValidationError('Feature in %s stage cannot be updated in %s environment.' % (self._feature_stage, server_mode.value))
        if server_mode == ServerMode.PROD and self._feature_stage in (ServerMode.DEV.value, ServerMode.TEST.value):
            raise utils.ValidationError('Feature in %s stage cannot be updated in %s environment.' % (self._feature_stage, server_mode.value))

    @classmethod
    def from_dict(cls, param_dict: PlatformParameterDict) -> PlatformParameter:
        if False:
            i = 10
            return i + 15
        'Returns an PlatformParameter object from a dict.\n\n        Args:\n            param_dict: dict. A dict mapping of all fields of\n                PlatformParameter object.\n\n        Returns:\n            PlatformParameter. The corresponding PlatformParameter domain\n            object.\n\n        Raises:\n            Exception. Given schema version is not supported.\n        '
        if param_dict['rule_schema_version'] != feconf.CURRENT_PLATFORM_PARAMETER_RULE_SCHEMA_VERSION:
            raise Exception("Current platform parameter rule schema version is v%s, received v%s, and there's no convert method from v%s to v%s." % (feconf.CURRENT_PLATFORM_PARAMETER_RULE_SCHEMA_VERSION, param_dict['rule_schema_version'], feconf.CURRENT_PLATFORM_PARAMETER_RULE_SCHEMA_VERSION, param_dict['rule_schema_version']))
        return cls(param_dict['name'], param_dict['description'], param_dict['data_type'], [PlatformParameterRule.from_dict(rule_dict) for rule_dict in param_dict['rules']], param_dict['rule_schema_version'], param_dict['default_value'], param_dict['is_feature'], param_dict['feature_stage'])

    def serialize(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the object serialized as a JSON string.\n\n        Returns:\n            str. JSON-encoded string encoding all of the information composing\n            the object.\n        '
        platform_parameter_dict = self.to_dict()
        return json.dumps(platform_parameter_dict)

    @classmethod
    def deserialize(cls, json_string: str) -> PlatformParameter:
        if False:
            return 10
        'Returns a PlatformParameter domain object decoded from a JSON\n        string.\n\n        Args:\n            json_string: str. A JSON-encoded string that can be\n                decoded into a dictionary representing a PlatformParameter.\n                Only call on strings that were created using serialize().\n\n        Returns:\n            PlatformParameter. The corresponding PlatformParameter domain\n            object.\n        '
        platform_parameter_dict = json.loads(json_string)
        platform_parameter = cls.from_dict(platform_parameter_dict)
        return platform_parameter